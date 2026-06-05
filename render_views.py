#!/usr/bin/env python3
"""Blender script: render multi-view images of a GLB model for 3DGS training.

Run: blender --background --python render_views.py

Outputs:
  {output_dir}/cameras.npz    - camera parameters
  {output_dir}/rgb_000.png    - rendered images
  {output_dir}/depth_000.png  - depth maps (optional, for validation)
"""

import bpy
import os
import sys
import json
import math
import numpy as np
from pathlib import Path
from mathutils import Matrix, Vector

# === DEFAULTS (override via command line) ===
RESOLUTION_X = 960
RESOLUTION_Y = 540
NUM_AZIMUTH = 50          # cameras per ring
NUM_ELEVATION = 2         # number of height rings
ELEVATION_ANGLES = [15, 45]  # degrees from horizontal plane
CAMERA_DISTANCE = 20.0    # meters from center
FOCAL_LENGTH_MM = 35.0    # lens focal length (35mm full-frame equivalent)
SENSOR_WIDTH_MM = 36.0
RENDER_SAMPLES = 1        # Only 1 sample needed for unlit/emission rendering


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_glb(filepath):
    bpy.ops.import_scene.gltf(filepath=filepath)
    print(f"Imported: {filepath}")
    # Collect imported objects
    objects = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not objects:
        objects = [o for o in bpy.data.objects if o.type == 'MESH']
    return objects


def compute_model_center(objects):
    """Compute bounding box center."""
    if not objects:
        return Vector((0, 0, 0))
    min_corner = Vector((float('inf'),) * 3)
    max_corner = Vector((float('-inf'),) * 3)
    for obj in objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world_corner.x)
            min_corner.y = min(min_corner.y, world_corner.y)
            min_corner.z = min(min_corner.z, world_corner.z)
            max_corner.x = max(max_corner.x, world_corner.x)
            max_corner.y = max(max_corner.y, world_corner.y)
            max_corner.z = max(max_corner.z, world_corner.z)
    center = (min_corner + max_corner) / 2
    size = max_corner - min_corner
    print(f"Model center: {center}, size: {size}")
    return center


def setup_unlit_materials():
    """Convert all materials to unlit/emission so rendered images match gsplat output.

    This replaces each material's shader with an emission shader that outputs
    the base color texture directly, bypassing all lighting calculations.
    """
    import bpy

    # Remove all lights
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Set world to black
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    bg = nodes.get('Background')
    if bg is None:
        bg = nodes.new('ShaderNodeBackground')
        out = nodes.get('World Output')
        if out:
            world.node_tree.links.new(bg.outputs['Background'], out.inputs['Surface'])
    bg.inputs['Color'].default_value = (0, 0, 0, 1)
    bg.inputs['Strength'].default_value = 0.0

    # For each material: replace with emission + texture
    for mat in list(bpy.data.materials):
        if not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Find the base color texture node
        tex_node = None
        bsdf_node = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf_node = node
            elif node.type == 'TEX_IMAGE':
                tex_node = node

        if tex_node is None or bsdf_node is None:
            continue

        # Create emission node
        emit = nodes.new('ShaderNodeEmission')
        emit.location = (bsdf_node.location.x, bsdf_node.location.y - 100)
        emit.inputs['Strength'].default_value = 1.0

        # Connect texture color → emission color
        links.new(tex_node.outputs['Color'], emit.inputs['Color'])

        # Connect emission → output
        out_node = None
        for node in nodes:
            if node.type == 'OUTPUT_MATERIAL':
                out_node = node
                break
        if out_node is None:
            out_node = nodes.new('ShaderNodeOutputMaterial')

        # Remove old BSDF links and connect emission
        for link in list(links):
            if link.to_node == out_node:
                links.remove(link)
        links.new(emit.outputs['Emission'], out_node.inputs['Surface'])

        # Set material to emit pass
        mat.pass_index = 1

    print("  Materials converted to unlit/emission")


def setup_render_engine():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.device = 'GPU'
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        for d in prefs.devices:
            d.use = (d.type == 'CUDA')
    except Exception:
        print("  Warning: Could not enable CUDA, falling back to CPU")

    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.film_transparent = True  # alpha background


def setup_camera(cam_obj, center, distance, azimuth_deg, elevation_deg):
    """Position camera on a sphere around the model center.

    azimuth: rotation around Z (horizontal), 0 = front
    elevation: angle above horizontal plane
    """
    az_rad = math.radians(azimuth_deg)
    el_rad = math.radians(elevation_deg)

    # Camera position on sphere
    x = center.x + distance * math.cos(el_rad) * math.cos(az_rad)
    y = center.y + distance * math.cos(el_rad) * math.sin(az_rad)
    z = center.z + distance * math.sin(el_rad)
    cam_obj.location = (x, y, z)

    # Look at center
    direction = Vector(center) - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Set focal length
    cam_obj.data.lens = FOCAL_LENGTH_MM
    cam_obj.data.sensor_width = SENSOR_WIDTH_MM


def compute_intrinsics(camera, resolution_x, resolution_y):
    """Compute camera intrinsic matrix K from Blender camera parameters."""
    # In Blender, the sensor fit might auto-adjust
    # Compute effective sensor dimensions
    sensor_width_mm = camera.data.sensor_width
    sensor_height_mm = camera.data.sensor_height

    # If auto sensor fit, adjust
    if camera.data.sensor_fit == 'AUTO':
        if resolution_x > resolution_y:
            sensor_height_mm = sensor_width_mm * resolution_y / resolution_x
        else:
            sensor_width_mm = sensor_height_mm * resolution_x / resolution_y
    elif camera.data.sensor_fit == 'VERTICAL':
        # sensor_height stays, adjust sensor_width
        sensor_width_mm = sensor_height_mm * resolution_x / resolution_y

    focal_length_mm = camera.data.lens

    fx = focal_length_mm * resolution_x / sensor_width_mm
    fy = focal_length_mm * resolution_y / sensor_height_mm
    cx = resolution_x / 2.0
    cy = resolution_y / 2.0

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)
    return K


def get_c2w(camera_obj):
    """Get camera-to-world 4x4 matrix."""
    return np.array(camera_obj.matrix_world, dtype=np.float64)


def main():
    # Parse command-line arguments (Blender ignores argparse, so use sys.argv)
    # Usage: blender --background --python render_views.py -- <glb_path> [output_dir]
    argv = sys.argv
    try:
        script_idx = argv.index('--python')
        args = argv[script_idx + 2:]  # after script name
    except (ValueError, IndexError):
        args = argv[argv.index('--') + 1:] if '--' in argv else argv[1:]

    if len(args) < 1:
        print("Usage: blender --background --python render_views.py -- <glb_path> [output_dir]")
        print("  glb_path    Path to input .glb model")
        print("  output_dir  Output directory (default: ./training_views)")
        sys.exit(1)

    glb_path = args[0]
    output_dir = args[1] if len(args) > 1 else './training_views'

    print("=" * 60)
    print("Blender Multi-View Renderer for 3DGS Training")
    print("=" * 60)
    print(f"  Input:  {glb_path}")
    print(f"  Output: {output_dir}")

    # Clear default scene
    clear_scene()

    # Import GLB
    objects = import_glb(glb_path)
    print(f"Loaded {len(objects)} mesh objects")

    # Compute center
    center = compute_model_center(objects)

    # Setup unlit materials (no lighting - matches gsplat output)
    setup_unlit_materials()

    # Setup render
    setup_render_engine()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create camera
    cam_data = bpy.data.cameras.new('RenderCam')
    cam_obj = bpy.data.objects.new('RenderCam', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Generate camera positions
    cameras_c2w = []
    cameras_K = []
    image_paths = []

    total_views = NUM_AZIMUTH * NUM_ELEVATION
    idx = 0

    for el_idx, elevation in enumerate(ELEVATION_ANGLES):
        for az_idx in range(NUM_AZIMUTH):
            azimuth = 360.0 * az_idx / NUM_AZIMUTH

            setup_camera(cam_obj, center, CAMERA_DISTANCE, azimuth, elevation)
            bpy.context.view_layer.update()

            # Get camera parameters
            c2w = get_c2w(cam_obj)
            K = compute_intrinsics(cam_obj, RESOLUTION_X, RESOLUTION_Y)

            # Render
            filename = f"rgb_{idx:04d}.png"
            filepath = os.path.join(output_dir, filename)
            bpy.context.scene.render.filepath = filepath
            bpy.ops.render.render(write_still=True)

            cameras_c2w.append(c2w)
            cameras_K.append(K)
            image_paths.append(filename)

            idx += 1
            if idx % 10 == 0:
                print(f"  Rendered {idx}/{total_views}")

    # Save camera parameters
    cameras_c2w = np.stack(cameras_c2w, axis=0)  # (N, 4, 4)
    cameras_K = np.stack(cameras_K, axis=0)      # (N, 3, 3)

    # Convert to W2C (world-to-camera) for 3DGS
    cameras_w2c = np.linalg.inv(cameras_c2w)

    np.savez(
        os.path.join(output_dir, 'cameras.npz'),
        c2w=cameras_c2w,
        w2c=cameras_w2c,
        K=cameras_K,
        image_paths=np.array(image_paths),
        resolution=np.array([RESOLUTION_X, RESOLUTION_Y]),
    )

    print(f"\nDone! Rendered {total_views} views to {output_dir}")
    print(f"  Resolution: {RESOLUTION_X}x{RESOLUTION_Y}")
    print(f"  Cameras saved to cameras.npz")


if __name__ == '__main__':
    main()
