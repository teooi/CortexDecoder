"""SDXL-base img2img conditioned on caption text embeddings (77x2048 tokens + 1280 pooled), from the DIP init."""
import json
import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..core.common import DEVICE, noop, torch_load_cpu
from .diffusion import _diffusion_paths, _load_ids, _part_path, final_recon_tag, select_best_candidates

SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_SIZE = 1024
OUT_SIZE = 512  # stored resolution; eval resizes to 256


def load_sdxl_img2img(cfg):
    from diffusers import StableDiffusionXLImg2ImgPipeline

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        SDXL_BASE,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir=f"{cfg.data_dir}/external_models/huggingface",
    ).to(DEVICE)
    pipe.vae.enable_slicing()  # the fp32-upcast VAE decode of a full batch at 1024px does not fit in 80GB
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_captions(cfg, img_ids):
    with open(f"{cfg.data_dir}/datasets/coco/nsd_captions.json") as f:
        captions = json.load(f)
    return [captions[str(int(i))][cfg.caption_index] for i in img_ids]


@torch.inference_mode()
def encode_captions(pipe, captions):
    """-> (prompt_embeds [B,77,2048], pooled [B,1280]). CFG negatives are SDXL's zeroed empty prompt."""
    embeds, _, pooled, _ = pipe.encode_prompt(
        prompt=captions, device=DEVICE, num_images_per_prompt=1, do_classifier_free_guidance=False,
    )
    return embeds.cpu(), pooled.cpu()


@torch.inference_mode()
def _generate(cfg, pipe, indices, embeds, pooled, lowlevel_images, seed):
    idx = torch.as_tensor(indices, dtype=torch.long)
    init = F.interpolate(lowlevel_images[idx].float(), size=(SDXL_SIZE, SDXL_SIZE), mode="bicubic", align_corners=False).clamp(0, 1)
    out = pipe(
        image=init.to(DEVICE, torch.float16),
        prompt_embeds=embeds[idx].to(DEVICE),
        pooled_prompt_embeds=pooled[idx].to(DEVICE),
        strength=(cfg.diffusion_num_steps - cfg.diffusion_start_step) / cfg.diffusion_num_steps,
        num_inference_steps=cfg.diffusion_num_steps,
        guidance_scale=cfg.sdxl_guidance_scale,
        generator=torch.Generator(device=DEVICE).manual_seed(seed),
        output_type="pt",
    ).images
    out = F.interpolate(out.float(), size=(OUT_SIZE, OUT_SIZE), mode="bicubic", align_corners=False)
    return out.clamp(0, 1).cpu().mul(255).round().byte()


def _generate_safe(cfg, pipe, indices, embeds, pooled, lowlevel_images, seed):
    try:
        return _generate(cfg, pipe, indices, embeds, pooled, lowlevel_images, seed)
    except torch.cuda.OutOfMemoryError:
        if len(indices) <= 1:
            raise
    # Retry outside the except block: inside it, the traceback still pins the failed attempt's tensors.
    torch.cuda.empty_cache()
    mid = len(indices) // 2
    return torch.cat([
        _generate_safe(cfg, pipe, indices[:mid], embeds, pooled, lowlevel_images, seed),
        _generate_safe(cfg, pipe, indices[mid:], embeds, pooled, lowlevel_images, seed + 1),
    ], dim=0)


def diffuse_text(cfg, pipe, data, dip_manifest, out_dir, on_batch_end=noop):
    """Same outputs/manifest as diffusion.diffuse, so evaluation.metrics.evaluate_final works unchanged."""
    tag = final_recon_tag(cfg)
    lowlevel_images = torch_load_cpu(dip_manifest["recons_path"]).float().clamp(0, 1)
    ids = _load_ids(dip_manifest["img_ids_path"])[:len(lowlevel_images)]
    if any(i not in data.final_heldout_ids for i in ids):
        raise ValueError("text diffusion inputs include non-heldout image IDs")
    captions = load_captions(cfg, ids)
    embeds, pooled = encode_captions(pipe, captions)
    print("text conditioning:", {"n": len(ids), "embeds": tuple(embeds.shape), "pooled": tuple(pooled.shape), "first": captions[:3]})

    paths = _diffusion_paths(cfg, out_dir, tag)
    n, k = len(ids), cfg.diffusion_ensemble_size
    pending = [i for i in range(n) if cfg.rerun_diffusion or not os.path.exists(_part_path(paths, i))]
    image_batch = max(1, cfg.sdxl_batch_size // k)
    for start in tqdm(range(0, len(pending), image_batch), desc="SDXL text img2img"):
        batch = pending[start:start + image_batch]
        candidates = _generate_safe(cfg, pipe, [i for i in batch for _ in range(k)], embeds, pooled, lowlevel_images, cfg.seed + start)
        samples, _, _ = select_best_candidates(cfg, batch, candidates, lowlevel_images)
        for i, sample in zip(batch, samples):
            torch.save(sample, _part_path(paths, i))
        on_batch_end()
    final = torch.stack([torch_load_cpu(_part_path(paths, i)) for i in range(n)]).contiguous()
    torch.save(final, paths["semantic_recons_path"])
    with open(paths["semantic_ids_path"], "w") as f:
        json.dump(ids, f)

    manifest = {
        "subject": cfg.subject, "n": n, "final_recon_tag": tag,
        "final_recons_path": paths["semantic_recons_path"], "final_ids_path": paths["semantic_ids_path"],
        "lowlevel_recons_path": dip_manifest["recons_path"], "lowlevel_ids_path": dip_manifest["img_ids_path"],
        **paths,
    }
    with open(f"{out_dir}/final_diffusion_manifest_{tag}.json", "w") as f:
        json.dump({cfg.subject: manifest}, f, indent=2)
    return manifest
