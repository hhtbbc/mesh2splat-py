# Mesh2Splat-py 🐍

A **headless Python pipeline** for converting 3D meshes (GLB) to 3D Gaussian Splatting (3DGS) format, with optional training and rendering. Designed for **Linux/Ubuntu servers** — no GUI, no OpenGL, no C++ compilation required.

Inspired by [EA's Mesh2Splat](https://github.com/electronicarts/mesh2splat) (C++/OpenGL, Windows-first), this project provides a pure Python, command-line-friendly alternative that runs on headless servers.

## ✨ Features

- **GLB → 3DGS PLY** — Convert textured 3D mesh to Gaussian Splatting point cloud in seconds
- **Headless by design** — No display, no OpenGL context, noxvfb hacks needed
- **Multi-view rendering** — Blender-based headless renderer for generating training views
- **3DGS training** — Optimize gaussians with [gsplat](https://github.com/nerfstudio-project/gsplat)'s differentiable rasterizer
- **Complete pipeline** — Go from a single `.glb` file to an optimized `.ply` ready for interactive viewing

## 📦 Installation

```bash
# Core dependencies (for GLB → PLY conversion)
pip install numpy Pillow trimesh

# Optional: training
pip install torch gsplat plyfile

# Optional: Blender for rendering training views
# Download Blender 4.2+ from https://www.blender.org/download/
```

## 🚀 Quick Start

### Step 1: Convert GLB to 3DGS PLY

```bash
python glb_to_3dgs_ply.py model.glb output.ply --samples 1000000
```

This samples 1M points on the mesh surface, extracts texture colors, and writes a standard 3DGS PLY file.

```
Options:
  --samples N     Number of gaussians (default: 1000000)
  --density F     Scale density factor (default: 0.5)
  --opacity F     Initial opacity (default: 0.9)
  --seed N        Random seed (default: 42)
```

### Step 2 (optional): Render multi-view training images

```bash
blender --background --python render_views.py -- model.glb ./training_views
```

Outputs `cameras.npz` + rendered RGBA images for training.

### Step 3 (optional): Train with gsplat

```bash
python train_3dgs.py \
  --input output.ply \
  --image_dir ./training_views \
  --cameras ./training_views/cameras.npz \
  --iterations 5000 \
  --output_dir ./training_output
```

Outputs `optimized.ply` — a refined 3DGS model ready for interactive viewing.

## 🔧 Pipeline Overview

```
┌──────────┐    ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐
│  GLB     │───▶│ glb_to_3dgs_ply  │───▶│  render_views │───▶│ train_3dgs   │
│  mesh    │    │ (Python, no GPU) │    │  (Blender)    │    │  (gsplat)    │
└──────────┘    └──────────────────┘    └──────────────┘    └──────────────┘
                      │                                             │
                      ▼                                             ▼
               initial.ply                                   optimized.ply
```

## 🆚 vs EA Mesh2Splat (C++)

| | EA Mesh2Splat | Mesh2Splat-py |
|---|---|---|
| Language | C++ (CMake) | Python |
| GUI dependency | GLFW + ImGui | None |
| OpenGL required | Yes | No |
| Headless Linux | Needs xvfb | Native |
| 3DGS training | Not included | gsplat integration |
| Multi-view render | Not included | Blender headless |
| Installation | Compile from source | `pip install` |

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🙏 Credits

- Inspired by [EA Mesh2Splat](https://github.com/electronicarts/mesh2splat) by Electronic Arts
- Built on [gsplat](https://github.com/nerfstudio-project/gsplat) and [trimesh](https://github.com/mikedh/trimesh)
- Blender for headless rendering
