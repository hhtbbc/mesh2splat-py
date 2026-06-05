#!/usr/bin/env python3
"""Train 3DGS gaussians from rendered multi-view images.

Initializes from an existing PLY and optimizes using gsplat's differentiable rasterizer.
Follows the original 3DGS paper training protocol.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData
from torch.utils.data import Dataset

from gsplat.rendering import rasterization


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

def load_ply(filepath, device='cuda'):
    """Load 3DGS PLY and return tensors."""
    ply = PlyData.read(filepath)
    vert = ply['vertex']
    n = vert.count

    means = torch.tensor(
        np.stack([vert['x'], vert['y'], vert['z']], axis=1),
        dtype=torch.float32, device=device,
    )
    quats = torch.tensor(
        np.stack([vert['rot_0'], vert['rot_1'], vert['rot_2'], vert['rot_3']], axis=1),
        dtype=torch.float32, device=device,
    )
    # Ensure unit quaternions
    quats = F.normalize(quats, dim=-1)

    scales = torch.tensor(
        np.stack([vert['scale_0'], vert['scale_1'], vert['scale_2']], axis=1),
        dtype=torch.float32, device=device,
    )
    opacities = torch.tensor(
        vert['opacity'], dtype=torch.float32, device=device,
    )

    # SH: DC (degree 0) as (N, 1, 3), rest (degrees 1-3) as (N, 15, 3)
    sh_dc = torch.tensor(
        np.stack([vert['f_dc_0'], vert['f_dc_1'], vert['f_dc_2']], axis=1),
        dtype=torch.float32, device=device,
    ).unsqueeze(1)  # (N, 1, 3)

    rest_keys = [f'f_rest_{i}' for i in range(45)]
    sh_rest = torch.tensor(
        np.stack([vert[k] for k in rest_keys], axis=1),
        dtype=torch.float32, device=device,
    ).reshape(n, 15, 3)  # (N, 15, 3)

    return means, quats, scales, opacities, sh_dc, sh_rest


def save_ply(filepath, means, quats, scales, opacities, sh_dc, sh_rest):
    """Save gaussians as 3DGS PLY."""
    import struct

    means_np = means.detach().cpu().numpy()
    quats_np = quats.detach().cpu().numpy()
    scales_np = scales.detach().cpu().numpy()
    opac_np = opacities.detach().cpu().numpy()
    sh_dc_np = sh_dc.detach().cpu().numpy()[:, 0, :]  # (N, 3)
    sh_rest_np = sh_rest.detach().cpu().numpy().reshape(-1, 45)  # (N, 45)

    n = means_np.shape[0]

    fields = []
    # Position & normal
    for name in ['x', 'y', 'z', 'nx', 'ny', 'nz']:
        arr = np.zeros(n, dtype=np.float32)
        if name == 'x': arr = means_np[:, 0]
        elif name == 'y': arr = means_np[:, 1]
        elif name == 'z': arr = means_np[:, 2]
        fields.append((name, arr))
    # SH DC
    for i, name in enumerate(['f_dc_0', 'f_dc_1', 'f_dc_2']):
        fields.append((name, sh_dc_np[:, i]))
    # SH rest
    for i in range(45):
        fields.append((f'f_rest_{i}', sh_rest_np[:, i]))
    # Opacity
    fields.append(('opacity', opac_np))
    # Scales
    for i, name in enumerate(['scale_0', 'scale_1', 'scale_2']):
        fields.append((name, scales_np[:, i]))
    # Rotation
    for i, name in enumerate(['rot_0', 'rot_1', 'rot_2', 'rot_3']):
        fields.append((name, quats_np[:, i]))

    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {n}']
    for name, _ in fields:
        header.append(f'property float {name}')
    header.append('end_header')

    with open(filepath, 'wb') as f:
        f.write('\n'.join(header).encode('ascii') + b'\n')
        for i in range(n):
            for _, arr in fields:
                f.write(struct.pack('<f', float(arr[i])))

    print(f"Saved {n} gaussians → {filepath}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RenderedDataset(Dataset):
    def __init__(self, image_dir, cameras_npz, resolution, device='cuda'):
        self.image_dir = Path(image_dir)
        self.device = device
        data = np.load(cameras_npz)
        self.w2c = torch.tensor(data['w2c'], dtype=torch.float32, device=device)
        self.K = torch.tensor(data['K'], dtype=torch.float32, device=device)
        self.paths = list(data['image_paths'])
        self.width, self.height = resolution

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_dir / self.paths[idx]), dtype=np.float32) / 255.0
        rgb = torch.tensor(img[..., :3], dtype=torch.float32, device=self.device)
        alpha = torch.tensor(
            img[..., 3:4] if img.shape[-1] == 4 else np.ones(img.shape[:-1] + (1,)),
            dtype=torch.float32, device=self.device,
        )
        return rgb, alpha, self.w2c[idx], self.K[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 20 * math.log10(1.0) - 10 * torch.log10(mse) if mse > 0 else torch.tensor(99.0)


def train(args):
    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # --- Dataset ---
    dataset = RenderedDataset(args.image_dir, args.cameras, (args.width, args.height), device)
    print(f"Dataset: {len(dataset)} views, {args.width}×{args.height}")

    # --- Load initial PLY ---
    print(f"Loading initial PLY: {args.input}")
    means, quats, scales, opacities, sh_dc, sh_rest = load_ply(args.input, device)
    n_init = means.shape[0]
    print(f"  {n_init} gaussians loaded")
    print(f"  Position range: X[{means[:,0].min():.1f},{means[:,0].max():.1f}] "
          f"Y[{means[:,1].min():.1f},{means[:,1].max():.1f}] "
          f"Z[{means[:,2].min():.1f},{means[:,2].max():.1f}]")

    # --- Scene scale ---
    scene_extent = max(
        means[:,0].max().item() - means[:,0].min().item(),
        means[:,1].max().item() - means[:,1].min().item(),
        means[:,2].max().item() - means[:,2].min().item(),
    )
    scene_scale = max(scene_extent, 1.0)
    print(f"  Scene scale: {scene_scale:.1f}")

    # --- Trainable parameters ---
    # We use torch.nn.Parameter so they are picked up by Adam correctly
    params = {
        'means': torch.nn.Parameter(means.clone()),
        'quats': torch.nn.Parameter(quats.clone()),
        'scales': torch.nn.Parameter(scales.clone()),
        'opacities': torch.nn.Parameter(opacities.clone()),
        'sh0': torch.nn.Parameter(sh_dc.clone()),
        'shN': torch.nn.Parameter(sh_rest.clone()),
    }

    # --- Optimizer (separate per parameter for strategy compatibility) ---
    lrs = {
        'means': args.lr_position,
        'quats': args.lr_rotation,
        'scales': args.lr_scale,
        'opacities': args.lr_opacity,
        'sh0': args.lr_sh,
        'shN': args.lr_sh / 20.0,
    }
    optimizers = {
        k: torch.optim.Adam([{'params': [params[k]], 'lr': lrs[k]}], eps=1e-15)
        for k in ['means', 'quats', 'scales', 'opacities', 'sh0', 'shN']
    }

    # --- Strategy (skip densification for now - gsplat 1.5.3 compat) ---
    # The initial gaussians from mesh sampling are already well-distributed.
    # We focus on optimizing SH, scales, opacities, and positions.

    # --- Training loop ---
    print(f"\n=== Training {args.iterations} steps ===")
    print(f"  SH degree: {args.sh_degree}")
    print(f"  LR: pos={args.lr_position} rot={args.lr_rotation} "
          f"scale={args.lr_scale} opa={args.lr_opacity} sh={args.lr_sh}")

    loss_ema = 0.0
    best_psnr = 0.0

    for step in range(1, args.iterations + 1):
        # Pick random view
        rgb_gt, alpha_gt, w2c, K = dataset[np.random.randint(0, len(dataset))]
        rgb_gt = rgb_gt  # (H, W, 3)
        alpha_gt = alpha_gt  # (H, W, 1)

        # Render
        opacities_act = torch.sigmoid(params['opacities'])
        scales_act = torch.exp(params['scales'])
        quats_norm = F.normalize(params['quats'], dim=-1)
        colors = torch.cat([params['sh0'], params['shN']], dim=1)  # (N, 16, 3)

        render_rgb, render_alpha, info = rasterization(
            means=params['means'],
            quats=quats_norm,
            scales=scales_act,
            opacities=opacities_act,
            colors=colors,
            viewmats=w2c.unsqueeze(0),   # (1, 4, 4)
            Ks=K.unsqueeze(0),           # (1, 3, 3)
            width=args.width,
            height=args.height,
            sh_degree=args.sh_degree,
            render_mode='RGB+D',
            eps2d=0.01,  # Allow tiny gaussians (default 0.3 would cull them)
            near_plane=0.001,
        )

        # Squeeze batch dimension and slice channels
        # RGB+D mode returns 4 channels (RGB + Depth)
        render_rgb = render_rgb.squeeze(0)[..., :3]    # (H, W, 3)
        render_alpha = render_alpha.squeeze(0)          # (H, W, 1)

        # Zero grads
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        # Loss
        # Only compute loss where the model is visible (alpha > 0.5)
        mask = (alpha_gt.squeeze(-1) > 0.5) & (render_alpha.squeeze(-1) > 0.01)
        if mask.sum() > 100:
            l1 = F.l1_loss(render_rgb[mask], rgb_gt[mask])
            alpha_l = F.l1_loss(render_alpha[mask], alpha_gt[mask])
        else:
            l1 = F.l1_loss(render_rgb, rgb_gt)
            alpha_l = F.l1_loss(render_alpha, alpha_gt)

        loss = l1 + 0.2 * alpha_l

        # Backward + step
        loss.backward()
        for opt in optimizers.values():
            opt.step()

        # --- Logging ---
        loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
        if step % 100 == 0 or step == 1:
            psnr_val = compute_psnr(render_rgb * alpha_gt + (1 - alpha_gt),
                                     rgb_gt * alpha_gt + (1 - alpha_gt))
            n_g = params['means'].shape[0]
            print(f"  [{step:6d}/{args.iterations}] loss={loss_ema:.6f} | "
                  f"PSNR={psnr_val.item():.2f} | gaussians={n_g:,} | "
                  f"lr={optimizers['means'].param_groups[0]['lr']:.2e}")

        # --- Checkpoint ---
        if step % 1000 == 0 or step == args.iterations:
            ckpt = os.path.join(args.output_dir, f'ckpt_{step:06d}.ply')
            save_ply(ckpt, params['means'], params['quats'], params['scales'],
                     params['opacities'], params['sh0'], params['shN'])

    # --- Final save ---
    final = os.path.join(args.output_dir, 'optimized.ply')
    save_ply(final, params['means'], params['quats'], params['scales'],
             params['opacities'], params['sh0'], params['shN'])

    print(f"\n✓ Training complete! Final gaussians: {params['means'].shape[0]:,}")
    print(f"  Output: {final}")
    return final


def main():
    p = argparse.ArgumentParser(description='Train 3DGS from rendered views')
    p.add_argument('--input', required=True, help='Input PLY')
    p.add_argument('--image_dir', required=True, help='Rendered image directory')
    p.add_argument('--cameras', required=True, help='cameras.npz file')
    p.add_argument('--output_dir', default='./training_output')
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--height', type=int, default=540)
    p.add_argument('--iterations', type=int, default=5000)
    p.add_argument('--sh_degree', type=int, default=3)
    p.add_argument('--lr_position', type=float, default=1.6e-4)
    p.add_argument('--lr_rotation', type=float, default=1e-3)
    p.add_argument('--lr_scale', type=float, default=5e-3)
    p.add_argument('--lr_opacity', type=float, default=5e-2)
    p.add_argument('--lr_sh', type=float, default=2.5e-3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
