"""
3D Gaussian Splatting scene representation.

Stores all learnable Gaussian parameters and provides:
- Initialization from a point cloud (Depth-Pro output)
- Differentiable rasterization via gsplat
- Adaptive density control (split/clone/prune) matching the original 3DGS paper
- Checkpoint save/load
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Spherical harmonics helpers
# ---------------------------------------------------------------------------

SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))

def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] to degree-0 SH coefficient."""
    return (rgb - 0.5) / SH_C0

def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    """Evaluate degree-0 SH → RGB [0,1]."""
    return sh * SH_C0 + 0.5


# ---------------------------------------------------------------------------
# GaussianScene
# ---------------------------------------------------------------------------

class GaussianScene(nn.Module):
    """
    Learnable 3D Gaussian scene.

    Parameters (all leaf tensors, optimised by Adam):
        _xyz        : (N, 3)  — Gaussian means
        _features_dc: (N, 1, 3) — degree-0 SH (colour)
        _features_rest:(N, D, 3) — higher-degree SH (view-dependent colour)
        _scaling    : (N, 3)  — log-scale
        _rotation   : (N, 4)  — quaternion (w, x, y, z)
        _opacity    : (N, 1)  — logit-opacity
    """

    def __init__(
        self,
        sh_degree: int = 3,
        device: str = "cuda",
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.active_sh_degree = 0          # ramped up during training
        self.device = device

        # Will be populated by init_from_pointcloud
        self._xyz: nn.Parameter | None = None
        self._features_dc: nn.Parameter | None = None
        self._features_rest: nn.Parameter | None = None
        self._scaling: nn.Parameter | None = None
        self._rotation: nn.Parameter | None = None
        self._opacity: nn.Parameter | None = None

        self._xyz_gradient_accum = None
        self._denom = None
        self.max_radii2D = None

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def init_from_pointcloud(
        self,
        xyz: torch.Tensor,         # (N, 3)
        rgb: Optional[torch.Tensor] = None,  # (N, 3) in [0, 1]
    ):
        """
        Seed Gaussians from a metric point cloud.
        Scales are initialised from nearest-neighbour distances (same as 3DGS paper).
        """
        N = xyz.shape[0]
        xyz = xyz.to(self.device)

        # Initial scale: mean distance to 3 nearest neighbours
        scales = self._estimate_initial_scales(xyz)  # (N, 3)

        # Rotations: identity quaternion (w=1, x=y=z=0)
        rots = torch.zeros(N, 4, device=self.device)
        rots[:, 0] = 1.0

        # Opacity: initialise to ~0.1 in sigmoid space
        opacities = torch.full((N, 1), -2.0, device=self.device)  # sigmoid(-2) ≈ 0.12

        # SH coefficients
        num_sh_rest = (self.sh_degree + 1) ** 2 - 1
        if rgb is not None:
            features_dc = rgb_to_sh(rgb.to(self.device)).unsqueeze(1)  # (N, 1, 3)
        else:
            features_dc = torch.zeros(N, 1, 3, device=self.device)
        features_rest = torch.zeros(N, num_sh_rest, 3, device=self.device)

        self._xyz           = nn.Parameter(xyz.float())
        self._features_dc   = nn.Parameter(features_dc.float())
        self._features_rest = nn.Parameter(features_rest.float())
        self._scaling       = nn.Parameter(torch.log(scales).float())
        self._rotation      = nn.Parameter(rots.float())
        self._opacity       = nn.Parameter(opacities.float())

        # Gradient accumulation buffers for adaptive density control
        self._xyz_gradient_accum = torch.zeros(N, 1, device=self.device)
        self._denom              = torch.zeros(N, 1, device=self.device)
        self.max_radii2D         = torch.zeros(N,    device=self.device)

        print(f"[GaussianScene] Initialised {N:,} Gaussians from point cloud.")

    def _estimate_initial_scales(self, xyz: torch.Tensor) -> torch.Tensor:
        """Estimate isotropic scale from mean kNN distance (k=3)."""
        # Brute-force kNN on GPU — fine for up to ~500k points
        dist2 = torch.cdist(xyz, xyz)
        dist2.fill_diagonal_(float("inf"))
        knn_dist, _ = dist2.topk(3, dim=1, largest=False)   # (N, 3)
        mean_dist = knn_dist.mean(dim=1, keepdim=True).sqrt()  # (N, 1)
        # Clamp to sane range
        mean_dist = mean_dist.clamp(1e-4, 1.0)
        return mean_dist.expand(-1, 3)  # isotropic

    # -----------------------------------------------------------------------
    # Accessors (activated values)
    # -----------------------------------------------------------------------

    @property
    def xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def rotation(self) -> torch.Tensor:
        """Normalised quaternion."""
        return F.normalize(self._rotation, dim=-1)

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def features(self) -> torch.Tensor:
        """All SH features concatenated: (N, (sh+1)^2, 3)."""
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0] if self._xyz is not None else 0

    # -----------------------------------------------------------------------
    # Rasterization (via gsplat)
    # -----------------------------------------------------------------------

    def render(
        self,
        camera_matrix: torch.Tensor,   # (4, 4) world-to-camera
        K: torch.Tensor,               # (3, 3) intrinsics
        H: int,
        W: int,
        bg_color: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Differentiable render via gsplat.

        Returns dict:
            rgb     : (H, W, 3)
            alpha   : (H, W, 1)
            depth   : (H, W, 1)
            radii   : (N,)  screen-space radii (for densification)
        """
        try:
            from gsplat import rasterization
        except ImportError:
            raise ImportError("gsplat not installed. Run: pip install gsplat")

        import torch.nn.functional as F

        if bg_color is None:
            bg_color = torch.ones(3, device=self.device)

        # gsplat expects (N, 3) means, (N, 4) quats, (N, 3) scales, (N,) opacities, (N, C) colours
        means3D  = self._xyz
        quats    = self.rotation
        scales   = self.scaling
        opacities= self.opacity.squeeze(-1)   # (N,)

        # Evaluate SH at degree 0 for simplicity (upgrade to view-dep later)
        colours = sh_to_rgb(self._features_dc.squeeze(1))  # (N, 3)
        colours = colours.clamp(0, 1)

        # Camera
        viewmat  = camera_matrix.unsqueeze(0)  # (1, 4, 4)
        Ks       = K.unsqueeze(0)              # (1, 3, 3)

        renders, alphas, meta = rasterization(
            means=means3D,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colours,
            viewmats=viewmat,
            Ks=Ks,
            width=W,
            height=H,
            backgrounds=bg_color.unsqueeze(0),
            render_mode="RGB+D",
        )

        rgb   = renders[0, ..., :3]   # (H, W, 3)
        depth = renders[0, ..., 3:4]  # (H, W, 1)
        alpha = alphas[0]             # (H, W, 1)

        return {"rgb": rgb, "alpha": alpha, "depth": depth, "meta": meta}

    # -----------------------------------------------------------------------
    # Adaptive density control
    # -----------------------------------------------------------------------

    def densify_and_prune(
        self,
        max_grad: float = 2e-4,
        min_opacity: float = 0.005,
        max_scale: float = 0.1,
        scene_extent: float = 1.0,
    ):
        """
        Clone under-reconstructed Gaussians, split over-large ones,
        prune transparent ones. Matches §3.3 of the original 3DGS paper.
        """
        grads = self._xyz_gradient_accum / self._denom.clamp(min=1)
        grads[grads.isnan()] = 0.0

        # Clone: small Gaussians with high gradient
        clone_mask = (grads.squeeze() >= max_grad) & \
                     (self.scaling.max(dim=-1).values <= 0.01 * scene_extent)

        # Split: large Gaussians with high gradient
        split_mask = (grads.squeeze() >= max_grad) & \
                     (self.scaling.max(dim=-1).values > 0.01 * scene_extent)

        # Prune: low opacity or too large
        prune_mask = (self.opacity.squeeze() < min_opacity) | \
                     (self.scaling.max(dim=-1).values > max_scale * scene_extent)

        self._clone_gaussians(clone_mask)
        self._split_gaussians(split_mask)
        self._prune_gaussians(prune_mask)

        # Reset accumulators
        N = self.num_gaussians
        self._xyz_gradient_accum = torch.zeros(N, 1, device=self.device)
        self._denom              = torch.zeros(N, 1, device=self.device)
        self.max_radii2D         = torch.zeros(N,    device=self.device)

        print(f"[Densify] {clone_mask.sum()} cloned, {split_mask.sum()} split, "
              f"{prune_mask.sum()} pruned → {self.num_gaussians:,} total")

    def _clone_gaussians(self, mask: torch.Tensor):
        if not mask.any():
            return
        self._xyz           = nn.Parameter(torch.cat([self._xyz,           self._xyz[mask]]))
        self._features_dc   = nn.Parameter(torch.cat([self._features_dc,   self._features_dc[mask]]))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, self._features_rest[mask]]))
        self._scaling       = nn.Parameter(torch.cat([self._scaling,       self._scaling[mask]]))
        self._rotation      = nn.Parameter(torch.cat([self._rotation,      self._rotation[mask]]))
        self._opacity       = nn.Parameter(torch.cat([self._opacity,       self._opacity[mask]]))

    def _split_gaussians(self, mask: torch.Tensor, N_split: int = 2):
        if not mask.any():
            return
        stdev = self.scaling[mask].repeat(N_split, 1)
        samples = torch.randn_like(stdev) * stdev
        new_xyz = self._xyz[mask].repeat(N_split, 1) + samples
        new_scale = nn.Parameter(
            torch.log(self.scaling[mask] / (0.8 * N_split)).repeat(N_split, 1)
        )
        # keep other params, add new ones
        keep = ~mask
        self._xyz           = nn.Parameter(torch.cat([self._xyz[keep],           new_xyz]))
        self._features_dc   = nn.Parameter(torch.cat([self._features_dc[keep],   self._features_dc[mask].repeat(N_split,1,1)]))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest[keep], self._features_rest[mask].repeat(N_split,1,1)]))
        self._scaling       = nn.Parameter(torch.cat([self._scaling[keep],       new_scale]))
        self._rotation      = nn.Parameter(torch.cat([self._rotation[keep],      self._rotation[mask].repeat(N_split,1)]))
        self._opacity       = nn.Parameter(torch.cat([self._opacity[keep],       self._opacity[mask].repeat(N_split,1)]))

    def _prune_gaussians(self, mask: torch.Tensor):
        keep = ~mask
        self._xyz           = nn.Parameter(self._xyz[keep])
        self._features_dc   = nn.Parameter(self._features_dc[keep])
        self._features_rest = nn.Parameter(self._features_rest[keep])
        self._scaling       = nn.Parameter(self._scaling[keep])
        self._rotation      = nn.Parameter(self._rotation[keep])
        self._opacity       = nn.Parameter(self._opacity[keep])

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "xyz":           self._xyz.data,
            "features_dc":   self._features_dc.data,
            "features_rest": self._features_rest.data,
            "scaling":       self._scaling.data,
            "rotation":      self._rotation.data,
            "opacity":       self._opacity.data,
            "sh_degree":     self.sh_degree,
        }, path)
        print(f"[GaussianScene] Saved {self.num_gaussians:,} Gaussians → {path}")

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.sh_degree      = ckpt["sh_degree"]
        self._xyz           = nn.Parameter(ckpt["xyz"].to(self.device))
        self._features_dc   = nn.Parameter(ckpt["features_dc"].to(self.device))
        self._features_rest = nn.Parameter(ckpt["features_rest"].to(self.device))
        self._scaling       = nn.Parameter(ckpt["scaling"].to(self.device))
        self._rotation      = nn.Parameter(ckpt["rotation"].to(self.device))
        self._opacity       = nn.Parameter(ckpt["opacity"].to(self.device))
        N = self._xyz.shape[0]
        self._xyz_gradient_accum = torch.zeros(N, 1, device=self.device)
        self._denom              = torch.zeros(N, 1, device=self.device)
        self.max_radii2D         = torch.zeros(N,    device=self.device)
        print(f"[GaussianScene] Loaded {N:,} Gaussians ← {path}")


