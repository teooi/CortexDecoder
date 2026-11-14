"""Decoder-aware loss: the frozen MindEye2 SDXL-unCLIP's own training loss, conditioned on the bridge's predicted tokens.

Same objective as unclip6.yaml StandardDiffusionLoss (LegacyDDPM discrete sigmas, offset noise 0.04, EpsWeighting), so the
gradient says which tokens let the frozen decoder denoise the true image. One UNet pass per batch, no sampling. Noise
levels are capped at `decoder_t_max`, the highest timestep our img2img sampling (start step 14 of 38) actually visits.
"""
import os

import numpy as np
import torch
from tqdm.auto import tqdm

from ..core.common import DEVICE, torch_load_cpu
from ..reconstruction.diffusion import load_combined_model

OFFSET_NOISE = 0.04


def legacy_ddpm_sigmas():
    """sigma for each of the 1000 discrete timesteps (index = timestep), as sgm DiscreteSampling."""
    betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float64) ** 2
    alphas = np.cumprod(1 - betas)
    return torch.tensor(((1 - alphas) / alphas) ** 0.5, dtype=torch.float32)


class UnclipDiffusionLoss:
    def __init__(self, cfg, data, stimuli):
        combined = load_combined_model(cfg)
        self.engine = combined.diffusion_engine
        self.vector = combined.get_vector_suffix().to(DEVICE).float()  # size/crop conditioning, as at sampling
        self.sigmas = legacy_ddpm_sigmas().to(DEVICE)
        self.t_max = int(cfg.decoder_t_max)
        self.latents = self._latents(cfg, data, stimuli)

    @torch.no_grad()
    def _latents(self, cfg, data, stimuli):
        """VAE latents of every stimulus the subject saw (256px, scaled), cached on the volume."""
        path = f"{cfg.data_dir}/unclip_latents_cache/{cfg.subject}_unclip_latents.pt"
        if os.path.exists(path):
            return torch_load_cpu(path)
        ids = sorted({int(i) for i in data.imgbrick_ids})
        latents = {}
        for start in tqdm(range(0, len(ids), 32), desc="unCLIP VAE latents", mininterval=30):
            batch = ids[start:start + 32]
            x = stimuli.for_ids(batch, size=256).to(DEVICE)
            z = self.engine.encode_first_stage(x * 2 - 1).float().cpu()
            latents.update({i: zi.half() for i, zi in zip(batch, z)})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(latents, path)
        return latents

    def __call__(self, pred_tokens, img_ids, generator=None):
        z0 = torch.stack([self.latents[int(i)] for i in img_ids]).to(DEVICE).float()
        n = z0.shape[0]
        t = torch.randint(0, self.t_max + 1, (n,), device=DEVICE, generator=generator)
        sigma = self.sigmas[t]
        noise = torch.randn(z0.shape, device=DEVICE, generator=generator)
        noise = noise + OFFSET_NOISE * torch.randn(n, 1, 1, 1, device=DEVICE, generator=generator)
        noised = z0 + noise * sigma[:, None, None, None]
        cond = {"crossattn": pred_tokens, "vector": self.vector.expand(n, -1)}
        denoised = self.engine.denoiser(self.engine.model, noised, sigma, cond).float()
        return ((denoised - z0) ** 2 / sigma[:, None, None, None] ** 2).mean()  # EpsWeighting: = eps-prediction MSE
