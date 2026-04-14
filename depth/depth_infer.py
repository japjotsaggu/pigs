"""
Depth-Pro inference wrapper.

Converts a single RGB image into a metric depth map + calibrated point cloud
that seeds the 3DGS initialization.

"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_depth_pro(device: str = "cuda") -> tuple:
    """
    Returns (model, transform) from the ml-depth-pro package.
    Lazy import so the rest of the project works even without the package.
    """
    try:
        from depth_pro.depth_pro import create_model_and_transforms
    except ImportError:
        raise ImportError(
            "ml-depth-pro not installed. "
            "Run: git clone https://github.com/apple/ml-depth-pro && pip install -e ml-depth-pro"
        )
    model, transform = create_model_and_transforms()
    model = model.to(device).eval()
    return model, transform


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def estimate_depth(
    image: Image.Image,
    model,
    transform,
    device: str = "cuda",
) -> dict:
    """
    Run Depth-Pro on a PIL image.

    Returns
    -------
    dict with keys:
        depth      : (H, W) float32 tensor — metric depth in metres
        focallength: float — estimated focal length in pixels
        intrinsics : (3, 3) float32 tensor — camera intrinsic matrix K
    """
    image_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)

    prediction = model.infer(image_tensor)

    depth = prediction["depth"].squeeze().cpu()          # (H, W)
    focallength_px = prediction["focallength_px"].item()

    W, H = image.size
    K = torch.tensor([
        [focallength_px, 0,              W / 2],
        [0,              focallength_px, H / 2],
        [0,              0,              1    ],
    ], dtype=torch.float32)

    return {"depth": depth, "focallength": focallength_px, "intrinsics": K}


# ---------------------------------------------------------------------------
# Point cloud lifting
# ---------------------------------------------------------------------------

def depth_to_pointcloud(
    depth: torch.Tensor,
    K: torch.Tensor,
    rgb: torch.Tensor | None = None,
    depth_min: float = 0.1,
    depth_max: float = 20.0,
) -> dict:
    """
    Back-project a depth map to a 3-D point cloud in camera space.

    Parameters
    ----------
    depth : (H, W) metric depth in metres
    K     : (3, 3) intrinsic matrix
    rgb   : (H, W, 3) uint8 or float32 image (optional, for colour)
    depth_min/max : clip range to discard sky / sensor noise

    Returns
    -------
    dict:
        xyz    : (N, 3) float32 — 3D positions in camera frame
        rgb    : (N, 3) float32 — colours in [0, 1], or None
        mask   : (H, W) bool   — valid pixel mask
    """
    H, W = depth.shape

    # Valid mask
    mask = (depth > depth_min) & (depth < depth_max)

    # Pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    # Unproject: X = (u - cx) * Z / fx
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    xyz = torch.stack([X, Y, Z], dim=-1)  # (H, W, 3)
    xyz_flat = xyz[mask]                  # (N, 3)

    out = {"xyz": xyz_flat, "mask": mask}

    if rgb is not None:
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)
        rgb_float = rgb.float() / 255.0 if rgb.max() > 1.0 else rgb.float()
        out["rgb"] = rgb_float[mask]
    else:
        out["rgb"] = None

    return out


# ---------------------------------------------------------------------------
# Convenience: image → point cloud in one call
# ---------------------------------------------------------------------------

def image_to_pointcloud(
    image_path: str | Path,
    model,
    transform,
    device: str = "cuda",
    **kwargs,
) -> dict:
    """
    Full pipeline: path → PIL → depth → point cloud.
    Returns the dict from depth_to_pointcloud plus the depth dict.
    """
    image = Image.open(image_path).convert("RGB")
    rgb_np = np.array(image)

    depth_out = estimate_depth(image, model, transform, device)

    pc = depth_to_pointcloud(
        depth=depth_out["depth"],
        K=depth_out["intrinsics"],
        rgb=torch.from_numpy(rgb_np),
        **kwargs,
    )
    pc["depth_map"] = depth_out["depth"]
    pc["intrinsics"] = depth_out["intrinsics"]
    pc["focallength"] = depth_out["focallength"]

    return pc


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_pointcloud_ply(xyz: torch.Tensor, rgb: torch.Tensor | None, path: str | Path):
    """Save as PLY for inspection in MeshLab / CloudCompare."""
    from plyfile import PlyData, PlyElement
    import numpy as np

    xyz_np = xyz.cpu().numpy()
    vertices = [tuple(row) for row in xyz_np]

    if rgb is not None:
        rgb_np = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                 ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        arr = np.array(
            [(*v, *c) for v, c in zip(xyz_np, rgb_np)],
            dtype=dtype,
        )
    else:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        arr = np.array(vertices, dtype=dtype)

    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(str(path))
    print(f"Saved point cloud → {path}  ({len(arr):,} points)")


# ---------------------------------------------------------------------------
# Quick test (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

    print("Loading Depth-Pro...")
    model, transform = load_depth_pro("cuda")

    print(f"Running on {img_path} ...")
    pc = image_to_pointcloud(img_path, model, transform, device="cuda")

    print(f"Point cloud: {pc['xyz'].shape[0]:,} points")
    print(f"Depth range: {pc['depth_map'].min():.2f}m – {pc['depth_map'].max():.2f}m")
    print(f"Focal length: {pc['focallength']:.1f}px")

    save_pointcloud_ply(pc["xyz"], pc["rgb"], "output_pointcloud.ply")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(img_path))
    plt.title("Input")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(pc["depth_map"].numpy(), cmap="plasma")
    plt.title("Metric depth (m)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("depth_output.png", dpi=150)
    print("Saved depth_output.png")