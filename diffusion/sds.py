"""
Score Distillation Sampling (SDS) loss using Zero123++ as the diffusion prior.

Zero123++ takes a single RGB image and generates multi-view consistent images
at specified camera poses. We use it as a score network: at each SDS step, we:
  1. Render the current Gaussians from a random novel viewpoint
  2. Add noise at a random timestep t
  3. Ask Zero123++ to predict the noise
  4. Use the score difference as a gradient signal (SDS)

References:
  - DreamFusion (Poole et al. 2022): https://arxiv.org/abs/2209.14988
  - Zero123++ (Shi et al. 2023): https://arxiv.org/abs/2310.15110
  - Stable Zero123: https://huggingface.co/stabilityai/stable-zero123
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SDSConfig:
    # Zero123++ model ID on HuggingFace
    model_id: str = "sudo-ai/zero123plus-v1.2"

    # SDS noise schedule
    t_min: float = 0.02
    t_max: float = 0.98

    # SDS guidance scale (classifier-free)
    guidance_scale: float = 7.5

    # Anneal t_max over training (SDS annealing from DreamFusion follow-ups)
    anneal_t_max: bool = True
    anneal_t_max_end: float = 0.5
    anneal_steps: int = 5000

    # Weight of SDS loss relative to photometric
    lambda_sds: float = 1.0

    device: str = "cuda"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_zero123pp(model_id: str = "sudo-ai/zero123plus-v1.2", device: str = "cuda"):
    """
    Load Zero123++ pipeline from HuggingFace.

    Returns the pipeline and its VAE/UNet for score extraction.
    """
    try:
        from diffusers import DiffusionPipeline, DDIMScheduler
    except ImportError:
        raise ImportError("Run: pip install diffusers transformers accelerate")

    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)

    # Use DDIM scheduler for deterministic noise stepping
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet.eval()
    pipeline.vae.eval()

    return pipeline


# ---------------------------------------------------------------------------
# SDS loss
# ---------------------------------------------------------------------------

class SDSLoss(torch.nn.Module):
    """
    Wraps Zero123++ to provide a differentiable SDS gradient signal.

    Usage:
        sds = SDSLoss(cfg)
        sds.load()   # loads the diffusion model

        # In the training loop:
        loss_sds = sds(
            rendered_rgb=rgb_render,    # (H, W, 3) float32
            cond_image=input_image,     # (H, W, 3) float32 — the reference view
            camera_pose=pose,           # (4, 4) world-to-camera
        )
    """

    def __init__(self, cfg: SDSConfig | None = None):
        super().__init__()
        self.cfg = cfg or SDSConfig()
        self.pipeline = None
        self._step_count = 0

    def load(self):
        self.pipeline = load_zero123pp(self.cfg.model_id, self.cfg.device)
        print(f"[SDS] Loaded Zero123++ from {self.cfg.model_id}")

    def _current_t_max(self) -> float:
        if not self.cfg.anneal_t_max:
            return self.cfg.t_max
        frac = min(self._step_count / self.cfg.anneal_steps, 1.0)
        return self.cfg.t_max - frac * (self.cfg.t_max - self.cfg.anneal_t_max_end)

    def encode_image(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Encode (H, W, 3) float32 image to VAE latent.
        """
        vae = self.pipeline.vae
        # (H, W, 3) → (1, 3, H, W), normalise to [-1, 1]
        x = rgb.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float16)
        x = x * 2.0 - 1.0

        # Resize to VAE input size (512x512)
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)

        with torch.no_grad():
            latent = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor

        return latent   # (1, 4, 64, 64)

    @torch.no_grad()
    def predict_noise(
        self,
        noisy_latent: torch.Tensor,   # (1, 4, 64, 64)
        t: torch.Tensor,              # scalar timestep
        cond_image: torch.Tensor,     # (H, W, 3) reference image
    ) -> torch.Tensor:
        """
        Run the UNet to predict noise at timestep t.
        Returns predicted noise (1, 4, 64, 64).
        """
        from PIL import Image
        import numpy as np

        unet      = self.pipeline.unet
        scheduler = self.pipeline.scheduler

        # Encode conditioning image for Zero123++
        cond_pil = Image.fromarray(
            (cond_image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        )
        # Zero123++ uses its own image encoder — get conditioning embedding
        cond_embeds = self.pipeline._encode_image(
            cond_pil,
            device=self.cfg.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        uncond_embeds, cond_embeds = cond_embeds.chunk(2)

        t_batch = t.reshape(1).to(self.cfg.device)

        # Classifier-free guidance
        latent_input = torch.cat([noisy_latent, noisy_latent])
        embeds_input = torch.cat([uncond_embeds, cond_embeds])

        noise_pred = unet(
            latent_input,
            t_batch.repeat(2),
            encoder_hidden_states=embeds_input,
        ).sample

        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + self.cfg.guidance_scale * (noise_cond - noise_uncond)

        return noise_pred

    def forward(
        self,
        rendered_rgb: torch.Tensor,     # (H, W, 3) — differentiable render
        cond_image: torch.Tensor,       # (H, W, 3) — reference image (detached)
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute SDS loss.

        SDS loss = E_t,ε [ w(t) * ||ε_φ(z_t; y, t) - ε||² ]
        where the gradient bypasses the UNet (stop_grad on noise prediction).

        Returns scalar loss.
        """
        if self.pipeline is None:
            raise RuntimeError("Call sds.load() before using SDSLoss")

        if step is not None:
            self._step_count = step
        else:
            self._step_count += 1

        t_max = self._current_t_max()

        # Sample random timestep
        t_val = torch.rand(1).item() * (t_max - self.cfg.t_min) + self.cfg.t_min
        t = torch.tensor(
            int(t_val * self.pipeline.scheduler.config.num_train_timesteps),
            dtype=torch.long,
            device=self.cfg.device,
        )

        # Encode rendered image to latent (with grad)
        latent = self.encode_image(rendered_rgb)   # (1, 4, 64, 64)

        # Sample noise
        noise = torch.randn_like(latent)

        # Add noise at timestep t
        scheduler = self.pipeline.scheduler
        noisy_latent = scheduler.add_noise(latent, noise, t.unsqueeze(0))

        # Predict noise (no grad through UNet)
        with torch.no_grad():
            noise_pred = self.predict_noise(
                noisy_latent.to(dtype=torch.float16),
                t,
                cond_image,
            )

        # SDS gradient: (noise_pred - noise), weighted by 1/alpha_t
        # Standard SDS loss (without Jacobian for efficiency)
        # w(t) = sigma_t^2 / alpha_t (from DreamFusion)
        alphas = scheduler.alphas_cumprod.to(self.cfg.device)
        alpha_t = alphas[t]
        w = 1.0 / (1.0 - alpha_t + 1e-8)

        grad = w * (noise_pred.float() - noise)
        grad = torch.nan_to_num(grad, nan=0.0)

        # SDS loss: grad is treated as a constant target (stop-gradient trick)
        # L_SDS = 0.5 * ||z - sg(z - grad)||^2
        # whose gradient w.r.t. z is exactly `grad`
        sds_loss = 0.5 * F.mse_loss(latent, (latent - grad).detach())

        return sds_loss * self.cfg.lambda_sds


# ---------------------------------------------------------------------------
# Camera sampling utilities (needed for SDS novel view sampling)
# ---------------------------------------------------------------------------

def sample_random_camera(
    radius: float = 2.5,
    theta_range: tuple = (30, 150),   # elevation in degrees (0=top, 90=side, 180=bottom)
    phi_range:   tuple = (0, 360),    # azimuth in degrees
    device: str = "cuda",
) -> torch.Tensor:
    """
    Sample a random camera pose (world-to-camera 4x4) on a sphere of given radius.
    Uses the same convention as Zero123++ (OpenCV camera).
    """
    theta = np.deg2rad(np.random.uniform(*theta_range))
    phi   = np.deg2rad(np.random.uniform(*phi_range))

    # Camera position in world frame
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.cos(theta)
    z = radius * np.sin(theta) * np.sin(phi)
    cam_pos = np.array([x, y, z])

    # Look-at: origin
    forward = -cam_pos / np.linalg.norm(cam_pos)
    up      = np.array([0, 1, 0])
    right   = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
    right  /= np.linalg.norm(right)
    up      = np.cross(right, forward)

    # World-to-camera rotation
    R = np.stack([right, up, -forward], axis=0)  # (3, 3)
    t = -R @ cam_pos                             # (3,)

    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3,  3] = cam_pos

    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3,  3] = t

    return torch.tensor(w2c, dtype=torch.float32, device=device)