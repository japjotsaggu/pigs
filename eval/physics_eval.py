"""
Physical plausibility evaluation harness.

The novel evaluation protocol:
  1. Export the trained Gaussian scene to a mesh (via marching cubes on
     the opacity field)
  2. Load the mesh into PyBullet as a static collision body
  3. Drop a rigid sphere from above
  4. Simulate for N steps
  5. Measure: did it behave realistically?

Metrics:
  - contact_time   : frames until first contact (lower = more grounded scene)
  - rest_height    : final height of sphere (should be > floor_height)
  - bounce_count   : number of bounces (physics sanity check)
  - penetration    : max penetration depth (lower = better geometry quality)
  - realistic_score: composite 0–1 score

Comparison:
  Run this on:
    - Our method (GGS + physics regularizer)
    - Baseline (GGS / DreamFusion without physics regularizer)
  A higher realistic_score = better physical plausibility.

Usage:
    python eval/physics_eval.py --checkpoint runs/exp_001/gaussians_final.pt \
                                 --output eval_results/
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    # Sphere properties
    sphere_radius:  float = 0.05    # metres
    sphere_mass:    float = 1.0     # kg
    sphere_restitution: float = 0.3 # bounciness

    # Drop parameters
    drop_height:    float = 1.0     # metres above scene centre
    drop_offset_xy: float = 0.0     # horizontal offset from centre

    # Simulation
    sim_steps:      int   = 500
    dt:             float = 1/240   # PyBullet default timestep

    # Mesh extraction
    voxel_res:      int   = 64      # marching cubes resolution
    opacity_thresh: float = 0.3     # iso-surface threshold

    # Scoring thresholds
    max_penetration_ok: float = 0.02  # 2cm max penetration for "good"

    device: str = "cpu"   # eval runs on CPU after Gaussian export


# ---------------------------------------------------------------------------
# Mesh extraction from Gaussians
# ---------------------------------------------------------------------------

def gaussians_to_mesh(
    xyz: torch.Tensor,        # (N, 3)
    opacity: torch.Tensor,    # (N, 1)
    scaling: torch.Tensor,    # (N, 3)
    cfg: EvalConfig,
    save_path: Optional[Path] = None,
):
    """
    Convert Gaussian splats to a watertight mesh for physics simulation.

    Strategy:
      1. Rasterise opacity field onto a 3D voxel grid by splatting each
         Gaussian as an isotropic Gaussian density kernel.
      2. Run marching cubes at the opacity threshold.
      3. Return a trimesh.Trimesh object.

    This is intentionally approximate — we need a collision mesh,
    not a perfect reconstruction.
    """
    try:
        import trimesh
        from skimage import measure
    except ImportError:
        raise ImportError("Run: pip install trimesh scikit-image")

    xyz_np    = xyz.cpu().numpy()
    op_np     = opacity.squeeze(-1).cpu().numpy()
    scale_np  = scaling.cpu().numpy().mean(axis=-1)  # isotropic mean scale

    R = cfg.voxel_res

    # Bounding box
    xyz_min = xyz_np.min(axis=0) - 0.1
    xyz_max = xyz_np.max(axis=0) + 0.1
    extent  = xyz_max - xyz_min

    # Voxel grid
    grid = np.zeros((R, R, R), dtype=np.float32)
    vox_size = extent / R

    # Splat each Gaussian onto the grid
    # For efficiency: only consider Gaussians with opacity > threshold
    vis_mask = op_np > 0.05
    xyz_vis  = xyz_np[vis_mask]
    op_vis   = op_np[vis_mask]
    sc_vis   = scale_np[vis_mask]

    # Grid coordinates of each Gaussian centre
    gc = ((xyz_vis - xyz_min) / extent * R).astype(np.int32)
    gc = np.clip(gc, 0, R-1)

    # Radius of influence in voxels
    r_vox = np.ceil(sc_vis / vox_size.mean() * 2).astype(np.int32).clip(1, 4)

    for i in range(len(xyz_vis)):
        cx, cy, cz = gc[i]
        r = r_vox[i]
        x0, x1 = max(0, cx-r), min(R, cx+r+1)
        y0, y1 = max(0, cy-r), min(R, cy+r+1)
        z0, z1 = max(0, cz-r), min(R, cz+r+1)

        xi = np.arange(x0, x1) - cx
        yi = np.arange(y0, y1) - cy
        zi = np.arange(z0, z1) - cz

        gx, gy, gz = np.meshgrid(xi, yi, zi, indexing="ij")
        d2 = (gx**2 + gy**2 + gz**2).astype(np.float32)
        sigma2 = max((sc_vis[i] / vox_size.mean()) ** 2, 0.5)
        kernel = op_vis[i] * np.exp(-0.5 * d2 / sigma2)

        grid[x0:x1, y0:y1, z0:z1] += kernel

    # Normalise
    grid /= (grid.max() + 1e-8)

    # Marching cubes
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            grid,
            level=cfg.opacity_thresh,
            spacing=(extent[0]/R, extent[1]/R, extent[2]/R),
        )
    except (ValueError, RuntimeError):
        print("[EvalWarning] Marching cubes failed — scene may be too sparse.")
        return None

    # Shift verts back to world coordinates
    verts += xyz_min

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=3)

    if save_path:
        mesh.export(str(save_path))
        print(f"[Eval] Mesh saved → {save_path}  ({len(verts)} verts, {len(faces)} faces)")

    return mesh


# ---------------------------------------------------------------------------
# PyBullet simulation
# ---------------------------------------------------------------------------

def run_drop_test(
    mesh,   # trimesh.Trimesh — the scene mesh
    cfg: EvalConfig,
) -> dict:
    """
    Drop a sphere onto the scene mesh and measure physical plausibility.

    Returns dict of metrics.
    """
    try:
        import pybullet as p
        import pybullet_data
        import trimesh
    except ImportError:
        raise ImportError("Run: pip install pybullet trimesh")

    import tempfile, os

    # ── Start PyBullet headless ──────────────────────────────────────────
    client = p.connect(p.DIRECT)   # headless
    p.setGravity(0, -9.81, 0, physicsClientId=client)
    p.setTimeStep(cfg.dt, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)

    # ── Load scene mesh as static body ───────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        mesh_path = f.name
    mesh.export(mesh_path)

    scene_col = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[1, 1, 1],
        physicsClientId=client,
    )
    scene_body = p.createMultiBody(
        baseMass=0,                    # static
        baseCollisionShapeIndex=scene_col,
        basePosition=[0, 0, 0],
        physicsClientId=client,
    )
    p.changeDynamics(scene_body, -1, restitution=0.3, physicsClientId=client)

    # ── Find scene floor height ──────────────────────────────────────────
    verts = np.array(mesh.vertices)
    floor_y = np.percentile(verts[:, 1], 5)   # 5th percentile Y = floor
    scene_centre = verts.mean(axis=0)

    # ── Drop sphere ──────────────────────────────────────────────────────
    drop_pos = [
        scene_centre[0] + cfg.drop_offset_xy,
        floor_y + cfg.drop_height,
        scene_centre[2] + cfg.drop_offset_xy,
    ]

    sphere_col = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=cfg.sphere_radius,
        physicsClientId=client,
    )
    sphere_body = p.createMultiBody(
        baseMass=cfg.sphere_mass,
        baseCollisionShapeIndex=sphere_col,
        basePosition=drop_pos,
        physicsClientId=client,
    )
    p.changeDynamics(
        sphere_body, -1,
        restitution=cfg.sphere_restitution,
        physicsClientId=client,
    )

    # ── Simulate ─────────────────────────────────────────────────────────
    positions = []
    contact_times = []
    max_penetration = 0.0
    bounce_count = 0
    prev_vy = 0.0

    for step in range(cfg.sim_steps):
        p.stepSimulation(physicsClientId=client)

        pos, _ = p.getBasePositionAndOrientation(sphere_body, physicsClientId=client)
        vel, _ = p.getBaseVelocity(sphere_body, physicsClientId=client)
        positions.append(pos)

        vy = vel[1]  # vertical velocity

        # Contact detection
        contacts = p.getContactPoints(sphere_body, scene_body, physicsClientId=client)
        if contacts:
            contact_times.append(step)
            for c in contacts:
                pen = -c[8]   # contact distance (negative = penetration)
                if pen > 0:
                    max_penetration = max(max_penetration, pen)

        # Bounce detection: vy switches sign while in contact
        if prev_vy < -0.05 and vy > 0.05 and contacts:
            bounce_count += 1
        prev_vy = vy

    # ── Cleanup ──────────────────────────────────────────────────────────
    p.disconnect(client)
    os.unlink(mesh_path)

    positions = np.array(positions)  # (T, 3)

    # ── Metrics ──────────────────────────────────────────────────────────
    first_contact = contact_times[0] if contact_times else cfg.sim_steps
    final_height  = float(positions[-1, 1])
    rest_height   = float(positions[-50:, 1].mean()) if len(positions) >= 50 else final_height

    # Did the sphere come to rest above the floor?
    resting_ok = rest_height > (floor_y - 0.02)   # within 2cm of floor

    # Penetration score
    penetration_ok = max_penetration < cfg.max_penetration_ok

    # Realistic score: composite
    # - Early contact = denser geometry (good)
    # - Low penetration = solid surfaces (good)
    # - Resting above floor = valid support (good)
    # - At least 1 bounce = physics is alive (sanity)
    contact_score     = 1.0 - min(first_contact / cfg.sim_steps, 1.0)
    penetration_score = 1.0 - min(max_penetration / 0.1, 1.0)
    resting_score     = 1.0 if resting_ok else 0.0
    bounce_score      = min(bounce_count / 2.0, 1.0)   # cap at 2 bounces

    realistic_score = (
        0.35 * contact_score +
        0.35 * penetration_score +
        0.20 * resting_score +
        0.10 * bounce_score
    )

    return {
        "contact_time_steps":  first_contact,
        "rest_height_m":       rest_height,
        "floor_height_m":      floor_y,
        "max_penetration_m":   max_penetration,
        "bounce_count":        bounce_count,
        "resting_ok":          resting_ok,
        "penetration_ok":      penetration_ok,
        "realistic_score":     realistic_score,
        # Sub-scores
        "contact_score":       contact_score,
        "penetration_score":   penetration_score,
        "resting_score":       resting_score,
        "bounce_score":        bounce_score,
        # Trajectory for visualisation
        "trajectory":          positions,
    }


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    cfg: EvalConfig | None = None,
    device: str = "cuda",
) -> dict:
    """
    Load a Gaussian checkpoint and run the full physical plausibility eval.
    """
    from gaussians.scene import GaussianScene

    cfg = cfg or EvalConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load scene
    scene = GaussianScene(device=device)
    scene.load(checkpoint_path)

    with torch.no_grad():
        xyz     = scene.xyz.cpu()
        opacity = scene.opacity.cpu()
        scaling = scene.scaling.cpu()

    print(f"[Eval] Loaded {scene.num_gaussians:,} Gaussians")

    # Extract mesh
    print("[Eval] Extracting mesh via opacity field...")
    mesh = gaussians_to_mesh(
        xyz, opacity, scaling, cfg,
        save_path=out / "scene_mesh.obj",
    )

    if mesh is None:
        print("[Eval] Mesh extraction failed — cannot run drop test.")
        return {"error": "mesh_extraction_failed"}

    # Run drop test
    print("[Eval] Running PyBullet drop test...")
    results = run_drop_test(mesh, cfg)

    # Print summary
    print("\n── Physical Plausibility Results ───────────────────")
    print(f"  Realistic score:   {results['realistic_score']:.3f}  (0=bad, 1=perfect)")
    print(f"  Contact time:      {results['contact_time_steps']} steps")
    print(f"  Max penetration:   {results['max_penetration_m']*100:.1f} cm "
          f"({'OK' if results['penetration_ok'] else 'HIGH'})")
    print(f"  Resting height:    {results['rest_height_m']:.3f} m "
          f"({'OK' if results['resting_ok'] else 'FAIL'})")
    print(f"  Bounce count:      {results['bounce_count']}")
    print("─────────────────────────────────────────────────────")

    # Save results
    import json
    save_results = {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in results.items()
                    if k != "trajectory"}
    with open(out / "eval_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n[Eval] Results saved → {out / 'eval_results.json'}")

    # Save trajectory
    np.save(out / "sphere_trajectory.npy", results["trajectory"])

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Physics plausibility evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to gaussians_final.pt")
    p.add_argument("--output",     default="eval_results/", help="Output directory")
    p.add_argument("--voxel-res",  type=int, default=64)
    p.add_argument("--sim-steps",  type=int, default=500)
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    cfg = EvalConfig(
        voxel_res=args.voxel_res,
        sim_steps=args.sim_steps,
    )

    evaluate_checkpoint(args.checkpoint, args.output, cfg, device=args.device)