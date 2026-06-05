#!/usr/bin/env python3
"""Convert GLB mesh to 3D Gaussian Splatting PLY format.

Samples points on mesh surfaces, extracts colors from textures,
and computes per-gaussian attributes (opacity, scale, rotation, SH coefficients).
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import struct

# SH constant: Y(0,0) = 1 / sqrt(4*pi)
C0 = 0.28209479177387814


def sample_texture_atlas(texture_pil, uvs):
    """Sample an atlas texture at given UV coordinates.

    Args:
        texture_pil: PIL Image (H x W x C)
        uvs: (N, 2) array of UV coordinates in [0, 1]

    Returns:
        (N, 3) float32 array of RGB colors in [0, 1]
    """
    tex_arr = np.asarray(texture_pil, dtype=np.float32) / 255.0
    h, w = tex_arr.shape[:2]

    # Clamp UVs
    u = np.clip(uvs[:, 0], 0.0, 1.0 - 1e-6)
    # Invert V (OpenGL texture convention: V=0 is bottom)
    v = np.clip(1.0 - uvs[:, 1], 0.0, 1.0 - 1e-6)

    px = (u * (w - 1)).astype(np.int32)
    py = (v * (h - 1)).astype(np.int32)

    colors = tex_arr[py, px, :3].astype(np.float32)
    return colors


def compute_scales(face_idx, face_areas, density_factor=0.5):
    """Compute per-gaussian scales based on local face area."""
    n_points = len(face_idx)
    unique_faces, counts = np.unique(face_idx, return_counts=True)
    face_to_count = dict(zip(unique_faces, counts))

    scales = np.zeros((n_points, 3), dtype=np.float32)
    for i in range(n_points):
        fi = face_idx[i]
        area = face_areas[fi]
        n_pts = max(face_to_count.get(fi, 1), 1)
        local_density = np.sqrt(area / n_pts)
        tangent_scale = local_density * density_factor
        normal_scale = local_density * density_factor * 0.3
        scales[i] = [tangent_scale, tangent_scale, normal_scale]
    return scales


def normal_to_quaternion(normals):
    """Convert surface normals to rotation quaternions (w, x, y, z)."""
    n = len(normals)
    quats = np.zeros((n, 4), dtype=np.float32)
    ref_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    for i in range(n):
        normal = normals[i]
        n_len = np.linalg.norm(normal)
        if n_len < 1e-8:
            quats[i] = [1.0, 0.0, 0.0, 0.0]
            continue
        normal = normal / n_len
        dot = np.dot(ref_z, normal)

        if dot > 0.99999:
            quats[i] = [1.0, 0.0, 0.0, 0.0]
        elif dot < -0.99999:
            quats[i] = [0.0, 1.0, 0.0, 0.0]
        else:
            axis = np.cross(ref_z, normal)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(dot)
            half_angle = angle / 2.0
            s = np.sin(half_angle)
            quats[i] = [np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s]
    return quats


def write_3dgs_ply(filepath, data):
    """Write a 3DGS PLY file (binary little-endian)."""
    n_points = len(data['x'])

    property_defs = [
        ('x', 'float'), ('y', 'float'), ('z', 'float'),
        ('nx', 'float'), ('ny', 'float'), ('nz', 'float'),
        ('f_dc_0', 'float'), ('f_dc_1', 'float'), ('f_dc_2', 'float'),
    ]
    for i in range(45):
        property_defs.append((f'f_rest_{i}', 'float'))
    property_defs += [
        ('opacity', 'float'),
        ('scale_0', 'float'), ('scale_1', 'float'), ('scale_2', 'float'),
        ('rot_0', 'float'), ('rot_1', 'float'), ('rot_2', 'float'), ('rot_3', 'float'),
    ]

    lines = ['ply', 'format binary_little_endian 1.0']
    lines.append(f'element vertex {n_points}')
    for name, dtype in property_defs:
        lines.append(f'property {dtype} {name}')
    lines.append('end_header')
    header = '\n'.join(lines) + '\n'

    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n_points):
            for name, _ in property_defs:
                f.write(struct.pack('<f', float(data[name][i])))

    print(f"Wrote {n_points} gaussians to {filepath}")


def convert_glb_to_3dgs(
    glb_path,
    output_path,
    total_samples=1_000_000,
    density_factor=0.5,
    opacity=0.9,
    seed=42,
):
    import trimesh

    rng = np.random.RandomState(seed)

    print(f"Loading {glb_path}...")
    scene = trimesh.load(glb_path)
    print(f"  Objects: {len(scene.geometry)}")

    # Merge all geometry
    merged = scene.to_geometry()
    print(f"  Merged mesh: {merged.vertices.shape[0]} vertices, {merged.faces.shape[0]} faces")

    if not isinstance(merged, trimesh.Trimesh):
        print("Error: merged geometry is not a Trimesh")
        return False

    mesh = merged

    # Get atlas texture from merged mesh
    atlas_texture = None
    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        atlas_texture = getattr(mesh.visual.material, 'baseColorTexture', None)
    if atlas_texture is not None:
        print(f"  Atlas texture: {atlas_texture.size[0]}x{atlas_texture.size[1]} {atlas_texture.mode}")
    else:
        print("  WARNING: No atlas texture found on merged mesh")

    # Compute face areas
    print("Computing face areas...")
    face_areas = mesh.area_faces
    total_area = face_areas.sum()
    print(f"  Total surface area: {total_area:.2f}")

    # Sample points uniformly by area
    print(f"Sampling {total_samples} points...")
    sample_face_probs = face_areas / total_area
    face_idx = rng.choice(len(mesh.faces), size=total_samples, p=sample_face_probs)

    # Barycentric coordinates
    r1 = rng.random(total_samples)
    r2 = rng.random(total_samples)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = r2 * sqrt_r1
    w = 1.0 - u - v
    bary = np.stack([u, v, w], axis=1)

    # World-space positions
    triangles = mesh.vertices[mesh.faces[face_idx]]
    points = np.einsum('ij,ijk->ik', bary, triangles)

    # Face normals
    face_normals = mesh.face_normals[face_idx]

    # Sample colors from atlas texture via UV interpolation
    colors = np.full((total_samples, 3), 0.5, dtype=np.float32)
    if atlas_texture is not None and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        print("  Sampling colors from atlas texture...")
        uvs_per_vertex = mesh.visual.uv
        face_uvs = uvs_per_vertex[mesh.faces[face_idx]]
        all_uvs = np.einsum('ij,ijk->ik', bary, face_uvs)
        colors = sample_texture_atlas(atlas_texture, all_uvs)
    else:
        print("  No texture/UV data, using default gray")

    # Convert sRGB -> linear (approximate gamma 2.2)
    colors_linear = np.power(np.clip(colors, 0.0, 1.0), 2.2)

    # SH DC coefficients: f_dc = color / C0
    # Because during rendering: color = C0 * f_dc
    sh_dc = colors_linear / C0

    # Compute scales (raw) then store as log(scale) — 3DGS convention
    print("Computing per-gaussian scales...")
    scales_raw = compute_scales(face_idx, face_areas, density_factor)
    scales = np.log(np.maximum(scales_raw, 1e-7))

    # Compute rotation quaternions from normals
    print("Computing rotation quaternions...")
    rotations = normal_to_quaternion(face_normals)

    # Opacity in logit space (inverse sigmoid) for PLY storage
    eps = 1e-8
    opac = np.clip(opacity, eps, 1.0 - eps)
    opacity_stored = np.log(opac / (1.0 - opac))

    # Assemble data
    data = {
        'x': points[:, 0].astype(np.float32),
        'y': points[:, 1].astype(np.float32),
        'z': points[:, 2].astype(np.float32),
        'nx': face_normals[:, 0].astype(np.float32),
        'ny': face_normals[:, 1].astype(np.float32),
        'nz': face_normals[:, 2].astype(np.float32),
        'f_dc_0': sh_dc[:, 0].astype(np.float32),
        'f_dc_1': sh_dc[:, 1].astype(np.float32),
        'f_dc_2': sh_dc[:, 2].astype(np.float32),
        'opacity': np.full(total_samples, opacity_stored, dtype=np.float32),
        'scale_0': scales[:, 0].astype(np.float32),
        'scale_1': scales[:, 1].astype(np.float32),
        'scale_2': scales[:, 2].astype(np.float32),
        'rot_0': rotations[:, 0].astype(np.float32),
        'rot_1': rotations[:, 1].astype(np.float32),
        'rot_2': rotations[:, 2].astype(np.float32),
        'rot_3': rotations[:, 3].astype(np.float32),
    }
    for i in range(45):
        data[f'f_rest_{i}'] = np.zeros(total_samples, dtype=np.float32)

    print(f"Writing {output_path}...")
    write_3dgs_ply(output_path, data)

    # Print color stats for verification
    print(f"\nColor stats (sRGB approx):")
    print(f"  R: [{colors[:,0].min():.3f}, {colors[:,0].max():.3f}] mean={colors[:,0].mean():.3f}")
    print(f"  G: [{colors[:,1].min():.3f}, {colors[:,1].max():.3f}] mean={colors[:,1].mean():.3f}")
    print(f"  B: [{colors[:,2].min():.3f}, {colors[:,2].max():.3f}] mean={colors[:,2].mean():.3f}")
    print(f"\nDone! Converted GLB -> 3DGS PLY with {total_samples} gaussians.")
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert GLB mesh to 3DGS PLY format')
    parser.add_argument('input', help='Input GLB file path')
    parser.add_argument('output', help='Output PLY file path')
    parser.add_argument('--samples', type=int, default=1_000_000,
                        help='Number of gaussians to sample (default: 1000000)')
    parser.add_argument('--density', type=float, default=0.5,
                        help='Scale density factor (default: 0.5)')
    parser.add_argument('--opacity', type=float, default=0.9,
                        help='Initial opacity (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()
    success = convert_glb_to_3dgs(
        glb_path=args.input, output_path=args.output,
        total_samples=args.samples, density_factor=args.density,
        opacity=args.opacity, seed=args.seed,
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
