"""SigLIP 2 unCLIP decoder (SDXL UNet fine-tuned on SigLIP2-giant patch tokens) behind BrainIT's `generate` interface.

Model loading follows code/extra/siglip2 finetune/brainit-siglip2-unclip-train.py preview(). Decoder settings copy the
decoder's own NSD evaluation (code/extra/modal_siglip2_eval.py, which gave PixCorr 0.84 on shared-1000 with true tokens):
CFG 5 with zero-token uncond, time_ids = NSD original size 425x425 / crop (0, 0), no contrast squash, and pure-noise
starts scaled by sqrt(1 + sigma_0^2). Img2img from a DIP image follows MindEye2's unclip_recon_new (noise + offset 0.04
at sigma[start_step]), the same path the CLIP decoder uses.
"""
import numpy as np
import torch
import torch.nn.functional as F

from ..core.common import DEVICE, require_file

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
BASE_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"  # = base_revision of production_siglip2_unclip_v4
GUIDANCE_SCALE = 5.0
OFFSET_NOISE = 0.04
TIME_IDS = (425.0, 425.0, 0.0, 0.0)  # NSD stimuli are 425x425; the decoder trained on [orig h, orig w, top, left]


def legacy_ddpm_schedule(num_steps):
    """SGM LegacyDDPMDiscretization: descending discrete timesteps and their sigmas (+ trailing 0)."""
    betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float64) ** 2
    alphas = np.cumprod(1 - betas)
    timesteps = np.linspace(999, 0, num_steps, endpoint=False).astype(int)
    sigmas = torch.tensor(((1 - alphas[timesteps]) / alphas[timesteps]) ** 0.5, dtype=torch.float32)
    return torch.as_tensor(timesteps), torch.cat([sigmas, sigmas.new_zeros(1)])


class SigLIP2UnCLIP:
    def __init__(self, cfg):
        from accelerate import init_empty_weights
        from diffusers import AutoencoderKL, UNet2DConditionModel

        path = require_file(f"{cfg.data_dir}/{cfg.siglip2_unclip_ckpt}")
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)  # 28 GB incl. optimizer
        print(f"SigLIP2 unCLIP: {path} step={state['global_step']} best_val={state['best_val']:.4f}")
        with init_empty_weights():
            unet = UNet2DConditionModel.from_config(state["unet_config"])
        unet.load_state_dict(state["model"], strict=True, assign=True)
        self.unet = unet.to(DEVICE, dtype=torch.bfloat16).eval().requires_grad_(False)
        del state
        self.vae = AutoencoderKL.from_pretrained(
            BASE_MODEL, subfolder="vae", variant="fp16", revision=BASE_REVISION,
            cache_dir=f"{cfg.data_dir}/external_models/huggingface", torch_dtype=torch.float32,
        ).to(DEVICE).eval().requires_grad_(False)
        self.scale = self.vae.config.scaling_factor

    @torch.no_grad()
    def generate(self, tokens, indexes=None, edge_index_v2c=None, img_init=None, num_samples=1, start_step=14, num_steps=38):
        n = tokens.shape[0]
        timesteps, sigmas = legacy_ddpm_schedule(num_steps)
        sigmas = sigmas.to(DEVICE)
        if img_init is None:
            start_step = 0
            x = torch.randn(n, 4, 32, 32, device=DEVICE) * torch.sqrt(1 + sigmas[0] ** 2)
        else:
            x = torch.from_numpy(img_init).permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=(256, 256), mode="bilinear").to(DEVICE)
            z = self.vae.encode(x * 2 - 1).latent_dist.sample() * self.scale
            noise = torch.randn_like(z) + OFFSET_NOISE * torch.randn(n, 1, 1, 1, device=DEVICE)
            x = z + noise * sigmas[start_step]
        cond = tokens.to(DEVICE, torch.bfloat16)
        context = torch.cat([torch.zeros_like(cond), cond])
        added = {"text_embeds": cond.new_empty((2 * n, 0)),
                 "time_ids": torch.tensor(TIME_IDS, device=DEVICE).repeat(2 * n, 1)}
        for i in range(start_step, num_steps):
            # DiscreteDenoiser + EpsScaling: denoised = x - sigma * eps, so the Euler derivative is eps.
            x_in = (x / torch.sqrt(1 + sigmas[i] ** 2)).to(torch.bfloat16)
            eps = self.unet(torch.cat([x_in, x_in]), timesteps[i].to(DEVICE), encoder_hidden_states=context,
                            added_cond_kwargs=added).sample.float()
            eps_u, eps_c = eps.chunk(2)
            x = x + (sigmas[i + 1] - sigmas[i]) * (eps_u + GUIDANCE_SCALE * (eps_c - eps_u))

        return ((self.vae.decode(x / self.scale).sample + 1) / 2).clamp(0, 1).cpu()

    def to(self, device):
        self.unet.to(device)
        self.vae.to(device)
        return self
