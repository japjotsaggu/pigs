# PIGS: Physically Informed Gaussian Scenes
### Physically Informed Gaussian Scenes for Geometry-Consistent 3D Generation

<p align="center">
  <img src="assets/teaser.gif" alt="PIGS teaser" width="720"/>
</p>

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"/></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1+-orange.svg" alt="PyTorch"/>
</p>

---

> **TL;DR:** State-of-the-art generative 3D scene models produce visually stunning results — but their geometry is physically broken. Furniture floats. Floors are hollow. Objects intersect. PIGS adds a differentiable physics regularizer on top of any Gaussian splatting backbone, enforcing gravity alignment, surface contact, and volumetric solidity — without sacrificing visual quality. We also introduce the first physics-based evaluation protocol for generative 3D scenes: drop a ball in, see if it rolls correctly.

---

## What's Wrong With Existing Methods?

Generative 3D scene models (GGS, LatentSplat, DreamFusion) are trained to optimize perceptual metrics — FID, LPIPS, PSNR. They produce scenes that *look* right from any viewpoint. But look under the hood:

- A table with no legs touching the floor
- A wall with Gaussians only on its visible face — hollow behind
- A chair floating 4cm above the ground plane

These failures are invisible to photometric losses. They only appear when you try to *use* the geometry — in robotics, simulation, or AR. **PIGS fixes this.**

---

## Method

<p align="center">
  <img src="assets/pipeline.png" alt="PIGS pipeline" width="720"/>
</p>

PIGS operates as a **physics regularization layer** on top of any 3D Gaussian Splatting backbone. Given a set of predicted Gaussian splats `{g^m}` with positions, scales, and opacities, we add three differentiable loss terms to the training objective:

### Physics Loss Terms

**Gravity Alignment Loss `L_g`**

Penalizes scene configurations where the opacity-weighted centroid of object Gaussian clusters lies above unsupported space. We estimate the dominant floor plane via RANSAC on the lowest-percentile Gaussians and enforce that object centroids project downward onto it.

```
L_g = Σ_i w_i · max(0, h_i - h_floor - ε)
```

where `w_i` is the opacity of Gaussian `i`, `h_i` is its height, and `h_floor` is the estimated floor plane.

**Surface Contact Loss `L_c`**

Penalizes floating objects by requiring that every connected component of Gaussians either (a) contacts the floor plane or (b) contacts another supported component. Implemented as a differentiable nearest-neighbor distance from each component's lowest point to its nearest support surface.

```
L_c = Σ_k max(0, d_contact(k) - δ)²
```

**Volumetric Solidity Loss `L_s`**

Penalizes hollow shells — Gaussian clusters that form closed surfaces with no interior density. We voxelize the Gaussian field and penalize low occupancy in the interior of convex hulls.

```
L_s = Σ_v max(0, ρ_threshold - ρ_interior(v))
```

### Total Objective

```
L_total = L_photometric + L_diffusion + λ_g·L_g + λ_c·L_c + λ_s·L_s
```

Default weights: `λ_g = 0.1`, `λ_c = 0.05`, `λ_s = 0.02`. These are tuned to preserve visual quality while enforcing physical plausibility.

---

## Novel Evaluation Protocol

Existing evaluations measure *visual* quality (FID, LPIPS). We introduce the first **physics-based evaluation** for generative 3D scenes:

1. Export generated Gaussian splats → triangle mesh (via marching cubes / Gaussian surfels)
2. Load mesh into **PyBullet** rigid-body simulator
3. Drop a sphere from height `h` above the scene
4. Measure:
   - **Contact rate**: does the sphere land on a surface (vs. fall through)?
   - **Support accuracy**: does it come to rest on the geometrically correct surface?
   - **Trajectory plausibility**: does the bounce/roll trajectory match physics expectations?

This evaluation is **model-agnostic** — run it on any method that produces a 3D scene representation.

---

## Results

> Results coming soon — table will be updated as experiments complete.

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.1+ with CUDA 11.8+
- 24GB VRAM recommended (tested on A100/RTX 3090)

### Installation

```bash
git clone https://github.com/yourusername/PIGS --recursive
cd PIGS

# Create environment
conda create -n pigs python=3.10
conda activate pigs

# Install dependencies
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Install diff-gaussian-rasterization
pip install ./submodules/diff-gaussian-rasterization

# Install physics evaluation dependencies
pip install pybullet trimesh
```

### Quick Start

```bash
# Run on a single image
python infer.py \
  --input assets/examples/living_room.jpg \
  --output output/scene \
  --physics_weight 0.1

# Visualize the generated scene
python visualize.py --scene output/scene

# Run physics evaluation (drop test)
python eval/drop_test.py --scene output/scene --visualize
```

### Training

```bash
# Train on RealEstate10K
python train.py \
  --dataset realestate10k \
  --data_root /path/to/realestate10k \
  --lambda_g 0.1 \
  --lambda_c 0.05 \
  --lambda_s 0.02 \
  --batch_size 4 \
  --output_dir checkpoints/pigs_re10k
```

---

## Repository Structure

```
PIGS/
├── pigs/
│   ├── physics/
│   │   ├── regularizer.py      # Core physics loss terms (L_g, L_c, L_s)
│   │   ├── floor_estimator.py  # RANSAC floor plane estimation
│   │   ├── component.py        # Connected component analysis on Gaussians
│   │   └── voxelizer.py        # Volumetric solidity computation
│   ├── backbone/
│   │   └── latentsplat.py      # LatentSplat wrapper / injection hook
│   ├── eval/
│   │   ├── drop_test.py        # PyBullet physics evaluation harness
│   │   ├── metrics.py          # Contact rate, support accuracy
│   │   └── export.py           # Gaussians → mesh for simulation
│   └── utils/
│       ├── gaussian_utils.py
│       └── visualization.py
├── train.py
├── infer.py
├── eval/
│   └── run_eval.py
├── configs/
│   ├── pigs_realestate10k.yaml
│   └── pigs_scannet.yaml
├── submodules/
│   └── diff-gaussian-rasterization/
├── assets/
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Description | Download |
|---|---|---|
| RealEstate10K | Indoor room videos, 10M frames | [Link](https://google.com/streetview/publish/special-terms) |
| ScanNet++ | High-res indoor scans | [Link](https://kaldir.vc.in.tum.de/scannetpp/) |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">Made with 🐷 and physics</p>
