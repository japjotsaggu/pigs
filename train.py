"""
Main training loop: physics-consistent 3DGS from a single image.

Pipeline:
  1. Load image → Depth-Pro → metric point cloud → seed GaussianScene
  2. For each iteration:
     a. Render from training viewpoints → photometric loss vs input views
     b. Render from random novel view → SDS loss via Zero123++
     c. Physics regularizer on Gaussian parameters
     d. Backprop, Adam step
     e. Every K steps: adaptive density control
  3. Save checkpoint → eval harness

Usage:
    python train.py --image data/living_room.jpg --output runs/exp_001
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


# ── project imports ─────────────────────────────────────────────────────────
from depth.depth_infer          import load_depth_pro, image_to_pointcloud
from gaussians.scene          import GaussianScene
from diffusion.sds            import SDSLoss, SDSConfig, sample_random_camera
from physics.regularizer      import PhysicsRegularizer, PhysicsConfig


# ---------------------------------------------------------------------------
# Optimiser factory — separate learning rates per parameter group
# ---------------------------------------------------------------------------

def build_optimiser(scene: GaussianScene, cfg: dict) -> torch.optim.Optimizer:
    lr = cfg.get("lr", {})
    param_groups = [
        {"params": [scene._xyz],           "lr": lr.get("xyz",      1.6e-4), "name": "xyz"},
        {"params": [scene._features_dc],   "lr": lr.get("feat_dc",  1e-3),   "name": "feat_dc"},
        {"params": [scene._features_rest], "lr": lr.get("feat_rest",5e-3),   "name": "feat_rest"},
        {"params": [scene._scaling],       "lr": lr.get("scaling",  5e-3),   "name": "scaling"},
        {"params": [scene._rotation],      "lr": lr.get("rotation", 1e-3),   "name": "rotation"},
        {"params": [scene._opacity],       "lr": lr.get("opacity",  5e-2),   "name": "opacity"},
    ]
    return torch.optim.Adam(param_groups, eps=1e-15)


# ---------------------------------------------------------------------------
# Photometric loss
# ---------------------------------------------------------------------------

def photometric_loss(
    rendered: torch.Tensor,     # (H, W, 3)
    target: torch.Tensor,       # (H, W, 3)
    lambda_l1: float = 0.8,
    lambda_ssim: float = 0.2,
) -> torch.Tensor:
    """L1 + SSIM loss matching the original 3DGS paper."""
    l1 = F.l1_loss(rendered, target)

    # SSIM: use a simple patch-based approximation
    ssim_val = ssim(rendered.permute(2,0,1).unsqueeze(0),
                    target.permute(2,0,1).unsqueeze(0))

    return lambda_l1 * l1 + lambda_ssim * (1.0 - ssim_val)


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural similarity — simplified single-scale version."""
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(pred,   window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    s1  = F.avg_pool2d(pred*pred,     window_size, stride=1, padding=window_size//2) - mu1_sq
    s2  = F.avg_pool2d(target*target, window_size, stride=1, padding=window_size//2) - mu2_sq
    s12 = F.avg_pool2d(pred*target,   window_size, stride=1, padding=window_size//2) - mu1_mu2
    num = (2*mu1_mu2 + C1) * (2*s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return (num / den.clamp(min=1e-8)).mean()


# ---------------------------------------------------------------------------
# Training camera schedule
# ---------------------------------------------------------------------------

def make_training_cameras(
    n_views: int,
    radius: float = 2.5,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """
    Generate a fixed set of training viewpoints around the scene.
    For the first view (index 0) we use the input camera (identity).
    """
    cameras = [torch.eye(4, device=device)]   # input view = identity
    for i in range(1, n_views):
        cameras.append(sample_random_camera(radius=radius, device=device))
    return cameras


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, args):
        self.args   = args
        self.device = args.device
        self.out    = Path(args.output)
        self.out.mkdir(parents=True, exist_ok=True)

        # Config
        self.n_iters          = args.iters
        self.densify_from     = 500
        self.densify_every    = 100
        self.densify_until    = 7_000   # was 15k — scaled to 10k run
        self.opacity_reset_every = 2_000   # was 3k
        self.sh_degree_every  = 800        # was 1k — still ramps SH through full degree

        self.use_sds     = args.sds
        self.use_physics = not args.no_physics

    def setup(self):
        """Load all models and initialise the Gaussian scene."""
        print("── Setup ───────────────────────────────────────────")

        # 1. Depth-Pro
        print("[1/4] Loading Depth-Pro...")
        self.depth_model, self.depth_transform = load_depth_pro(self.device)

        # 2. Point cloud → Gaussians
        print("[2/4] Running depth estimation...")
        self.input_image = Image.open(self.args.image).convert("RGB")
        pc = image_to_pointcloud(
            self.args.image,
            self.depth_model,
            self.depth_transform,
            device=self.device,
        )
        print(f"      {pc['xyz'].shape[0]:,} points, "
              f"depth {pc['depth_map'].min():.2f}–{pc['depth_map'].max():.2f}m")

        # Subsample point cloud if too large
        max_points = 100_000
        if pc['xyz'].shape[0] > max_points:
            idx = torch.randperm(pc['xyz'].shape[0])[:max_points]
            pc['xyz'] = pc['xyz'][idx]
            if pc['rgb'] is not None:
                pc['rgb'] = pc['rgb'][idx]
            print(f"      Subsampled to {max_points:,} points")

        self.intrinsics = pc["intrinsics"].to(self.device)
        W_img, H_img   = self.input_image.size

        # Downscale large images — 800px wide is plenty for 3DGS optimization
        self.scale = min(1.0, 800.0 / max(W_img, H_img))
        self.H = int(H_img * self.scale)
        self.W = int(W_img * self.scale)

        # Scale intrinsics to match render resolution
        if self.scale < 1.0:
            self.intrinsics = self.intrinsics.clone()
            self.intrinsics[0, 0] *= self.scale   # fx
            self.intrinsics[1, 1] *= self.scale   # fy
            self.intrinsics[0, 2] *= self.scale   # cx
            self.intrinsics[1, 2] *= self.scale   # cy

        # Resize target image to match render resolution
        import torch.nn.functional as F
        target_np = np.array(self.input_image.resize((self.W, self.H),
                             Image.BILINEAR)).astype(np.float32) / 255.0
        self.target_rgb = torch.from_numpy(target_np).to(self.device)

        print(f"      Render resolution: {self.W}×{self.H} (scale={self.scale:.2f})")

        # 3. Init Gaussian scene
        print("[3/4] Initialising Gaussian scene...")
        self.scene = GaussianScene(sh_degree=3, device=self.device)
        self.scene.init_from_pointcloud(pc["xyz"], pc["rgb"])

        # 4. SDS + physics
        if self.use_sds:
            print("[4/4] Loading Zero123++ for SDS...")
            self.sds = SDSLoss(SDSConfig(device=self.device))
            self.sds.load()
        else:
            print("[4/4] SDS disabled (--no-sds)")
            self.sds = None

        self.physics_reg = PhysicsRegularizer(PhysicsConfig()) if self.use_physics else None

        # Optimiser
        self.optimiser = build_optimiser(self.scene, {
            "lr": {
                "xyz":       1e-4,
                "feat_dc":   1e-3,
                "feat_rest": 5e-3,
                "scaling":   5e-3,
                "rotation":  1e-3,
                "opacity":   5e-2,
            }
        })

        # Training cameras (input view + random novel views for photometric supervision)
        self.train_cameras = make_training_cameras(
            n_views=1,   # just input view for photometric; novel views via SDS
            radius=float(pc["depth_map"].mean().item()),
            device=self.device,
        )

        print("── Setup complete ──────────────────────────────────\n")

    def train(self):
        import wandb
        if self.args.wandb:
            wandb.init(project="physics-3dgs", config=vars(self.args))

        print(f"Training for {self.n_iters} iterations...")
        t0 = time.time()

        for step in range(self.n_iters):
            self.optimiser.zero_grad()

            # ── Render input view ──────────────────────────────
            render = self.scene.render(
                camera_matrix=self.train_cameras[0],
                K=self.intrinsics,
                H=self.H,
                W=self.W,
            )
            rgb_render = render["rgb"]   # (H, W, 3)

            # Ensure target matches render size exactly
            if self.target_rgb.shape[:2] != rgb_render.shape[:2]:
                self.target_rgb = F.interpolate(
                    self.target_rgb.permute(2,0,1).unsqueeze(0),
                    size=rgb_render.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1,2,0)

            if step == 0:
                import torchvision
                torchvision.utils.save_image(rgb_render.permute(2,0,1).clamp(0,1), "debug_render_step0.png")
                torchvision.utils.save_image(self.target_rgb.permute(2,0,1).clamp(0,1), "debug_target.png")
                xyz = self.scene.xyz
                op  = self.scene.opacity
                sc  = self.scene.scaling
                print(f"\n[DEBUG] xyz: X={xyz[:,0].min():.3f}~{xyz[:,0].max():.3f} "
                      f"Y={xyz[:,1].min():.3f}~{xyz[:,1].max():.3f} "
                      f"Z={xyz[:,2].min():.3f}~{xyz[:,2].max():.3f}")
                print(f"[DEBUG] opacity: min={op.min():.4f} max={op.max():.4f} mean={op.mean():.4f}")
                print(f"[DEBUG] scaling: min={sc.min():.6f} max={sc.max():.6f} mean={sc.mean():.6f}")
                print(f"[DEBUG] alpha mean: {render['alpha'].mean().item():.6f}")
                print(f"[DEBUG] N visible (op>0.05): {(op.squeeze()>0.05).sum().item()}")
                print(f"[DEBUG] N in front (Z>0): {(xyz[:,2]>0).sum().item()}\n")

            # ── Photometric loss ───────────────────────────────
            loss_photo = photometric_loss(rgb_render, self.target_rgb)

            # ── SDS loss (novel view) ──────────────────────────
            loss_sds = torch.zeros(1, device=self.device)
            if self.use_sds and self.sds is not None:
                # Sample a random novel camera and render
                novel_cam = sample_random_camera(device=self.device)
                novel_render = self.scene.render(
                    camera_matrix=novel_cam,
                    K=self.intrinsics,
                    H=256, W=256,   # smaller for SDS to keep memory reasonable
                )
                loss_sds = self.sds(
                    rendered_rgb=novel_render["rgb"],
                    cond_image=F.interpolate(
                        self.target_rgb.permute(2,0,1).unsqueeze(0),
                        size=(256, 256), mode="bilinear"
                    ).squeeze(0).permute(1,2,0),
                    step=step,
                )

            # ── Physics regularizer ────────────────────────────
            physics_losses = {}
            loss_physics = torch.zeros(1, device=self.device)
            if self.use_physics and self.physics_reg is not None:
                # Ramp in physics loss after initial warmup
                # 300 steps = enough for Gaussians to find rough geometry first
                if step > 300:
                    physics_losses = self.physics_reg(
                        xyz=self.scene.xyz,
                        scaling=self.scene.scaling,
                        opacity=self.scene.opacity,
                    )
                    loss_physics = physics_losses["loss_physics"]

            # ── Total loss ─────────────────────────────────────
            loss = loss_photo + loss_sds + loss_physics
            loss.backward()

            if step % 50 == 0:
                grad = self.scene._xyz.grad
                print(f"  [grad] xyz: {grad.abs().mean().item():.2e}" if grad is not None else "  [grad] xyz: None")

            self.optimiser.step()

            # Exponential decay of xyz lr (3DGS paper: 1.6e-4 → 1.6e-6 over 30k steps)
            # Scaled for our 10k run
            for group in self.optimiser.param_groups:
                if group.get("name") == "xyz":
                    t = step / self.n_iters
                    group["lr"] = 1e-4 * (1e-2 ** t)  # decays ~100x over full run

            # ── Adaptive density control ───────────────────────
            if (self.densify_from <= step < self.densify_until
                    and step % self.densify_every == 0):
                # Accumulate gradient norms for densification criterion
                if self.scene._xyz.grad is not None:
                    grad_norm = self.scene._xyz.grad.norm(dim=-1, keepdim=True)
                    self.scene._xyz_gradient_accum += grad_norm
                    self.scene._denom += 1
                self.scene.densify_and_prune(
                    scene_extent=float(self.scene.scaling.mean().item()) * 100
                )
                # Rebuild optimiser (parameters changed after densify)
                self.optimiser = build_optimiser(self.scene, {})

            # Periodically reset low-opacity Gaussians
            if step > 0 and step % self.opacity_reset_every == 0:
                with torch.no_grad():
                    self.scene._opacity.data.clamp_(max=0.01)

            # Gradually increase SH degree
            if step > 0 and step % self.sh_degree_every == 0:
                self.scene.active_sh_degree = min(
                    self.scene.active_sh_degree + 1,
                    self.scene.sh_degree,
                )

            # ── Logging ────────────────────────────────────────
            if step % 100 == 0:
                elapsed = time.time() - t0
                log = {
                    "step":          step,
                    "loss/total":    loss.item(),
                    "loss/photo":    loss_photo.item(),
                    "loss/sds":      loss_sds.item() if self.use_sds else 0,
                    "loss/physics":  loss_physics.item() if self.use_physics else 0,
                    "n_gaussians":   self.scene.num_gaussians,
                    "elapsed_s":     elapsed,
                }
                if physics_losses:
                    log["loss/gravity"]  = physics_losses.get("loss_gravity",  0).item()
                    log["loss/contact"]  = physics_losses.get("loss_contact",  0).item()
                    log["loss/solidity"] = physics_losses.get("loss_solidity", 0).item()

                print(f"  step {step:5d} | photo={log['loss/photo']:.4f} "
                      f"sds={log['loss/sds']:.4f} "
                      f"physics={log['loss/physics']:.4f} "
                      f"N={log['n_gaussians']:,}")

                if self.args.wandb:
                    wandb.log(log, step=step)

            # ── Checkpoint ─────────────────────────────────────
            if step > 0 and step % 5000 == 0:
                ckpt_path = self.out / f"gaussians_{step:06d}.pt"
                self.scene.save(ckpt_path)

        # Final save
        self.scene.save(self.out / "gaussians_final.pt")
        print(f"\nDone. Final checkpoint → {self.out / 'gaussians_final.pt'}")

        if self.args.wandb:
            wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Physics-consistent 3DGS trainer")
    p.add_argument("--image",      required=True,          help="Input RGB image path")
    p.add_argument("--output",     default="runs/exp_001", help="Output directory")
    p.add_argument("--iters",      type=int, default=10_000, help="Optimisation iterations (default 10k ≈ 45min on A40)")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--sds",        action="store_true",    help="Enable SDS via Zero123++ (adds ~4h, needs 48GB)")
    p.add_argument("--no-physics", action="store_true",    help="Disable physics regularizer (baseline run)")
    p.add_argument("--wandb",      action="store_true",    help="Log to Weights & Biases")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()