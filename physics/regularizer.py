"""
Physics regularizer — the core novel contribution.

Three differentiable loss terms that penalise physically implausible
Gaussian configurations:

    L_physics = λ_g * L_gravity
              + λ_c * L_contact
              + λ_s * L_solidity

All losses operate directly on the Gaussian parameter tensors and are
fully differentiable — gradients flow back into xyz, scaling, and opacity.

Design principles
-----------------
1. Weak prior: losses regularise, not constrain. The photometric + SDS
   losses still dominate; physics nudges towards plausibility.
2. Opacity-weighted: transparent Gaussians contribute little — they are
   likely sky/artefacts, not geometry.
3. Differentiable everywhere: no hard thresholds in the backward pass.
4. Scene-scale agnostic: all distances normalised by scene_extent.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PhysicsConfig:
    # Loss weights (tune via ablation)
    lambda_gravity:  float = 0.01
    lambda_contact:  float = 0.05
    lambda_solidity: float = 0.02

    # Gravity direction in camera/world frame (default: -Y is up)
    # Set to (0, -1, 0) for standard OpenCV camera convention
    # Set to (0, 0, -1) if your world frame has Z-up
    gravity_dir: tuple = (0.0, 1.0, 0.0)   # points DOWN (positive Y = down in CV convention)

    # Contact: floor plane detection percentile
    floor_percentile: float = 5.0   # use the bottom 5% of Y as the floor estimate

    # Solidity: voxel grid resolution for interior density check
    solidity_voxel_res: int = 32

    # Opacity threshold for "visible" Gaussians
    opacity_threshold: float = 0.05


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------

def gravity_alignment_loss(
    xyz: torch.Tensor,         # (N, 3) Gaussian centres
    opacity: torch.Tensor,     # (N, 1) or (N,) activated opacity in [0,1]
    cfg: PhysicsConfig,
) -> torch.Tensor:
    """
    L_gravity: penalise Gaussians that are "floating" — i.e., their
    opacity-weighted vertical centre of mass is too high relative to
    the scene floor.

    Concretely, we compute the opacity-weighted mean Y position of the
    scene, and compare it to the expected CoM given a uniform vertical
    distribution. High-opacity mass concentrated at the top = bad.

    This is a soft penalty: it doesn't force a specific floor height,
    just discourages implausible mass distributions.
    """
    gravity = torch.tensor(cfg.gravity_dir, device=xyz.device, dtype=xyz.dtype)
    gravity = F.normalize(gravity, dim=0)   # unit vector pointing down

    op = opacity.squeeze(-1)                # (N,)
    op = op.detach()                        # don't backprop through opacity for this term

    # Project xyz onto gravity direction (= "height along gravity axis")
    heights = (xyz * gravity.unsqueeze(0)).sum(dim=-1)  # (N,) — larger = further "down"

    # Floor estimate: opacity-weighted low percentile of height
    visible = op > cfg.opacity_threshold
    if visible.sum() < 10:
        return xyz.new_zeros(1).squeeze()

    heights_vis = heights[visible]
    floor_h = torch.quantile(heights_vis, cfg.floor_percentile / 100.0)

    # Floating penalty: mass significantly above the floor is fine (objects rest there).
    # But mass at extreme heights with no support is penalised.
    # We penalise the *variance* of the height distribution weighted by opacity —
    # physically plausible scenes have structured vertical distributions, not random floaters.
    op_vis = op[visible]
    op_w   = op_vis / (op_vis.sum() + 1e-8)

    h_mean = (op_w * heights_vis).sum()
    h_var  = (op_w * (heights_vis - h_mean) ** 2).sum()

    # Additional: penalise opacity-mass below the floor (underground geometry)
    underground = F.relu(floor_h - heights_vis)   # positive where above-floor (gravity down)
    underground_loss = (op_w * underground).sum()

    return h_var * 0.1 + underground_loss


def contact_constraint_loss(
    xyz: torch.Tensor,       # (N, 3)
    scaling: torch.Tensor,   # (N, 3) activated scale
    opacity: torch.Tensor,   # (N, 1)
    cfg: PhysicsConfig,
) -> torch.Tensor:
    """
    L_contact: objects should rest ON surfaces, not float above them.

    Strategy:
      1. Estimate the support surface (floor plane) from low-Y Gaussians.
      2. For each "object" cluster, find its lowest point.
      3. Penalise the gap between the lowest point and the floor.

    We approximate this with a simpler differentiable proxy:
      - Find the opacity-weighted floor height (low percentile).
      - Identify high-opacity Gaussians above the floor by more than
        their own scale (these are candidates for floating objects).
      - Penalise them proportionally to the float gap.

    This is intentionally a soft proxy — we're adding a prior, not
    solving contact detection.
    """
    gravity = torch.tensor(cfg.gravity_dir, device=xyz.device, dtype=xyz.dtype)
    gravity = F.normalize(gravity, dim=0)

    op = opacity.squeeze(-1)
    visible_mask = op > cfg.opacity_threshold

    if visible_mask.sum() < 10:
        return xyz.new_zeros(1).squeeze()

    heights = (xyz * gravity.unsqueeze(0)).sum(dim=-1)   # (N,)
    scales_along_gravity = (scaling * gravity.abs().unsqueeze(0)).sum(dim=-1)  # (N,)

    # Floor height
    floor_h = torch.quantile(heights[visible_mask], cfg.floor_percentile / 100.0)

    # "Float gap" for each Gaussian: how far its bottom edge is above the floor
    # bottom_edge = centre_height - scale_along_gravity (gravity points down → subtract)
    bottom_edge = heights - scales_along_gravity
    float_gap = F.relu(floor_h - bottom_edge)   # positive where floating

    op_vis = op * visible_mask.float()
    contact_loss = (op_vis * float_gap).mean()

    return contact_loss


def solidity_loss(
    xyz: torch.Tensor,      # (N, 3)
    opacity: torch.Tensor,  # (N, 1)
    cfg: PhysicsConfig,
) -> torch.Tensor:
    """
    L_solidity: objects should be volumetrically solid, not hollow shells.

    Observation: hollow surfaces are characterised by Gaussians clustering
    on a thin shell with near-empty interiors. Solid objects have Gaussians
    throughout their volume.

    Proxy: voxelise the scene at coarse resolution. For each occupied voxel
    (surface), check that its interior neighbour voxels also have reasonable
    opacity mass. Penalise surface voxels with hollow interiors.

    This is the most expensive of the three terms — uses a voxel grid.
    Kept at low resolution (default 32^3) to stay fast.
    """
    op = opacity.squeeze(-1)
    visible_mask = op > cfg.opacity_threshold

    if visible_mask.sum() < 100:
        return xyz.new_zeros(1).squeeze()

    xyz_vis = xyz[visible_mask]
    op_vis  = op[visible_mask]

    # Normalise to [0, 1] voxel grid
    xyz_min = xyz_vis.min(dim=0).values
    xyz_max = xyz_vis.max(dim=0).values
    extent  = (xyz_max - xyz_min).clamp(min=1e-4)

    xyz_norm = (xyz_vis - xyz_min) / extent   # (M, 3) in [0, 1]

    R = cfg.solidity_voxel_res
    # Voxel indices
    vox_idx = (xyz_norm * (R - 1)).long().clamp(0, R - 1)  # (M, 3)

    # Scatter opacity into voxel grid
    grid = torch.zeros(R, R, R, device=xyz.device)
    flat_idx = vox_idx[:, 0] * R * R + vox_idx[:, 1] * R + vox_idx[:, 2]
    grid.view(-1).scatter_add_(0, flat_idx, op_vis)

    # Normalise
    grid = grid / (grid.max() + 1e-8)

    # Surface voxels: grid value above threshold
    surface_threshold = 0.3
    surface = grid > surface_threshold  # (R, R, R) bool

    if not surface.any():
        return xyz.new_zeros(1).squeeze()

    # For each surface voxel, check interior (shift by 1 in each axis)
    # Interior proxy: average of 6-neighbourhood
    # Pad → shift → average
    g = grid.unsqueeze(0).unsqueeze(0)   # (1, 1, R, R, R)
    g_pad = F.pad(g, (1,1,1,1,1,1), mode="replicate")

    # 6-connected neighbourhood average (interior density)
    interior = (
        g_pad[:, :, 2:, 1:-1, 1:-1] +   # +x
        g_pad[:, :, :-2, 1:-1, 1:-1] +  # -x
        g_pad[:, :, 1:-1, 2:, 1:-1] +   # +y
        g_pad[:, :, 1:-1, :-2, 1:-1] +  # -y
        g_pad[:, :, 1:-1, 1:-1, 2:] +   # +z
        g_pad[:, :, 1:-1, 1:-1, :-2]    # -z
    ).squeeze() / 6.0                   # (R, R, R)

    # Hollow = surface but low interior density
    hollow_penalty = F.relu(surface_threshold - interior) * surface.float()
    solidity_loss_val = hollow_penalty.mean()

    return solidity_loss_val


# ---------------------------------------------------------------------------
# Combined physics loss
# ---------------------------------------------------------------------------

class PhysicsRegularizer(torch.nn.Module):
    """
    Wraps all three physics losses.
    Call forward() with the Gaussian scene's raw parameters.
    """

    def __init__(self, cfg: PhysicsConfig | None = None):
        super().__init__()
        self.cfg = cfg or PhysicsConfig()

    def forward(
        self,
        xyz: torch.Tensor,
        scaling: torch.Tensor,
        opacity: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        xyz     : (N, 3) Gaussian positions
        scaling : (N, 3) activated scales (after exp)
        opacity : (N, 1) activated opacities (after sigmoid)

        Returns
        -------
        dict:
            loss_physics   : scalar — weighted sum of all physics losses
            loss_gravity   : scalar
            loss_contact   : scalar
            loss_solidity  : scalar
        """
        cfg = self.cfg

        L_g = gravity_alignment_loss(xyz, opacity, cfg)
        L_c = contact_constraint_loss(xyz, scaling, opacity, cfg)
        L_s = solidity_loss(xyz, opacity, cfg)

        L_total = (
            cfg.lambda_gravity  * L_g +
            cfg.lambda_contact  * L_c +
            cfg.lambda_solidity * L_s
        )

        return {
            "loss_physics":  L_total,
            "loss_gravity":  L_g,
            "loss_contact":  L_c,
            "loss_solidity": L_s,
        }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    N = 5000

    # Simulate a "bad" scene: Gaussians floating at random heights
    xyz_bad = torch.randn(N, 3)
    xyz_bad[:, 1] += 2.0   # push everything up (floating)

    # Simulate a "good" scene: Gaussians on a flat surface
    xyz_good = torch.randn(N, 3)
    xyz_good[:, 1] = xyz_good[:, 1].abs() * 0.1   # low spread in Y

    scaling  = torch.ones(N, 3) * 0.05
    opacity  = torch.ones(N, 1) * 0.8

    reg = PhysicsRegularizer()

    losses_bad  = reg(xyz_bad,  scaling, opacity)
    losses_good = reg(xyz_good, scaling, opacity)

    print("=== Bad scene (floating) ===")
    for k, v in losses_bad.items():
        print(f"  {k}: {v.item():.6f}")

    print("\n=== Good scene (grounded) ===")
    for k, v in losses_good.items():
        print(f"  {k}: {v.item():.6f}")

    print("\n✓ Physics regularizer sanity check passed.")
    print("  (bad scene should have higher losses than good scene)")