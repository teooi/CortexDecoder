"""BrainIT/MindEye2 SDXL-unCLIP diffusion: low-level DIP image init + predicted CLIP-token conditioning."""
import importlib
import json
import logging
import os
import sys
import types
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..core.common import DEVICE, noop, require_file, torch_load_cpu
from ..core.config import CLIP_NUM_QUERIES
from ..data.nsd import rows_for_ids


def brainit_paths(cfg):
    brainit_dir = f"{cfg.data_dir}/brainit-fmri"
    external = f"{brainit_dir}/data/external_models"
    saved = f"{brainit_dir}/results/saved_models"
    mindeye2_dir = os.environ.get("MINDEYE2_MODEL_DIR", f"{external}/MindEyeV2")
    return {
        "brainit_dir": brainit_dir,
        "unclip_ckpt": os.environ.get("MINDEYE2_UNCLIP_CKPT_PATH", f"{mindeye2_dir}/unclip6_epoch0_step110000.ckpt"),
        "unclip_config": f"{brainit_dir}/src/MindEyeV2/generative_models/configs/unclip6.yaml",
        "sdxl_config": f"{brainit_dir}/src/MindEyeV2/generative_models/configs/inference/sd_xl_base.yaml",
        "sdxl_ckpt": os.environ.get("BRAINIT_SDXL_CKPT_PATH", f"{external}/SDXL/zavychromaxl_v30.safetensors"),
        "combined_ckpt": os.environ.get("BRAINIT_COMBINED_CKPT_PATH", f"{saved}/combined_model.ckpt"),
        "clipg_model": os.environ.get("BRAINIT_CLIPG_MODEL_PATH", f"{saved}/decoder_clipg_ext-1_save.pth"),
    }


def _import_brainit(brainit_dir):
    """Match BrainIT's import layout: repo root must beat src/MindEyeV2/utils.py and src/MindEyeV2/models.py."""
    os.chdir(brainit_dir)
    paths = [brainit_dir, f"{brainit_dir}/src", f"{brainit_dir}/src/MindEyeV2", f"{brainit_dir}/src/MindEyeV2/generative_models"]
    sys.path[:] = [p for p in sys.path if p not in paths]
    sys.path[:0] = paths
    for name in list(sys.modules):
        root = name.split(".")[0]
        if root in {"utils", "models", "open_clip", "generative_models", "sgm"} or name.startswith("src.MindEyeV2.generative_models"):
            del sys.modules[name]
    importlib.invalidate_caches()

    import models as brainit_models_pkg
    import utils as brainit_utils_pkg
    for pkg, sub in [(brainit_utils_pkg, "utils"), (brainit_models_pkg, "models")]:
        expected = os.path.abspath(f"{brainit_dir}/{sub}")
        if not any(os.path.abspath(p).startswith(expected) for p in getattr(pkg, "__path__", [])):
            raise RuntimeError(f"Wrong {sub} import: {getattr(pkg, '__file__', pkg)}")

    # clip-anytorch still imports pkg_resources.packaging.
    import packaging
    try:
        import pkg_resources
    except ImportError:
        pkg_resources = types.ModuleType("pkg_resources")
        sys.modules["pkg_resources"] = pkg_resources
    if not hasattr(pkg_resources, "packaging"):
        pkg_resources.packaging = packaging
    import clip  # noqa: F401
    import timm  # noqa: F401
    import torch_geometric  # noqa: F401

    try:
        from diffusers.models.vae import Decoder  # noqa: F401
    except Exception:
        from diffusers.models.autoencoders.vae import Decoder
        vae_compat = types.ModuleType("diffusers.models.vae")
        vae_compat.Decoder = Decoder
        sys.modules["diffusers.models.vae"] = vae_compat


def _patch_openclip_text_encoder_batch_first(sgm_encoder_modules):
    def is_batch_first(model_):
        transformer = getattr(model_, "transformer", None)
        if bool(getattr(transformer, "batch_first", False)):
            return True
        blocks = getattr(transformer, "resblocks", [])
        return bool(blocks and getattr(getattr(blocks[0], "attn", None), "batch_first", False))

    def forward_blocks(self, x, attn_mask=None):
        outputs = {}
        batch_first = is_batch_first(self.model)
        blocks = self.model.transformer.resblocks
        for i, block in enumerate(blocks):
            if i == len(blocks) - 1:
                outputs["penultimate"] = x if batch_first else x.permute(1, 0, 2)
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = torch.utils.checkpoint.checkpoint(block, x, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask=attn_mask)
        outputs["last"] = x if batch_first else x.permute(1, 0, 2)
        return outputs

    def encode2(self, text):
        x = self.model.token_embedding(text) + self.model.positional_embedding
        if not is_batch_first(self.model):
            x = x.permute(1, 0, 2)
        outputs = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            return self.model.ln_final(outputs[self.layer])
        last = self.model.ln_final(outputs["last"])
        outputs["pooled"] = last[torch.arange(last.shape[0], device=last.device), text.argmax(dim=-1)] @ self.model.text_projection
        outputs["last"] = last
        return outputs

    def encode1(self, text):
        x = self.model.token_embedding(text) + self.model.positional_embedding
        if not is_batch_first(self.model):
            x = x.permute(1, 0, 2)
        for i, block in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = torch.utils.checkpoint.checkpoint(block, x, self.model.attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask=self.model.attn_mask)
        if not is_batch_first(self.model):
            x = x.permute(1, 0, 2)
        return self.model.ln_final(x)

    sgm_encoder_modules.FrozenOpenCLIPEmbedder2.text_transformer_forward = forward_blocks
    sgm_encoder_modules.FrozenOpenCLIPEmbedder2.encode_with_transformer = encode2
    sgm_encoder_modules.FrozenOpenCLIPEmbedder.encode_with_transformer = encode1


class PrecomputedClipAdapter(nn.Module):
    """Replaces BrainIT's fMRI->CLIP GNN: our bridge already predicts the CLIP tokens."""

    def forward(self, fmri, indexes=None, edge_index_v2c=None):
        return fmri


def _disable_inference_checkpointing(engine):
    disabled = False
    model_ = getattr(engine, "model", None)
    for obj in [engine, model_, getattr(model_, "diffusion_model", None)]:
        fn = getattr(obj, "disable_gradient_checkpointing", None) if obj is not None else None
        if callable(fn):
            fn()
            disabled = True
    return disabled


def load_combined_model(cfg):
    if DEVICE.type != "cuda":
        raise RuntimeError("BrainIT diffusion uses CUDA-only calls.")
    paths = brainit_paths(cfg)
    for key in ["unclip_config", "unclip_ckpt", "combined_ckpt", "clipg_model"]:
        require_file(paths[key])
    if cfg.run_brainit_sdxl_enhancement:
        require_file(paths["sdxl_config"])
        require_file(paths["sdxl_ckpt"])

    warnings.filterwarnings("ignore", message=r"`torch\.cuda\.amp\.autocast.*deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=r"torch\.utils\.checkpoint: the use_reentrant parameter should be passed explicitly.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"None of the inputs have requires_grad=True.*", category=UserWarning)

    import xformers.ops
    q = torch.randn(1, 8, 64, 64, device=DEVICE, dtype=torch.float16)
    xformers.ops.memory_efficient_attention(q, q, q)

    _import_brainit(paths["brainit_dir"])
    logging.getLogger("sgm.modules.attention").setLevel(logging.ERROR)
    from generative_models.sgm.modules.encoders import modules as sgm_encoder_modules
    from models.combined_diffusion_engine import CombinedDiffusionEngine
    from utils.diffusion_utils import load_diffusion_engine
    _patch_openclip_text_encoder_batch_first(sgm_encoder_modules)

    print("loading MindEye2 unCLIP diffusion engine...")
    base_engine = load_diffusion_engine(config_path=paths["unclip_config"], ckpt_path=paths["unclip_ckpt"])
    print("gradient checkpointing disabled:", _disable_inference_checkpointing(base_engine))
    print("loading BrainIT combined diffusion checkpoint...")
    combined = CombinedDiffusionEngine.load_from_checkpoint(
        paths["combined_ckpt"],
        map_location="cpu",
        weights_only=False,
        diffusion_engine=base_engine,
        gnn_model=torch_load_cpu(paths["clipg_model"]),
    )
    combined.gnn_model = PrecomputedClipAdapter()
    combined = combined.to(DEVICE).eval().requires_grad_(False)
    combined.diffusion_engine.sampler.num_steps = cfg.diffusion_num_steps
    print("diffusion ready; BrainIT SDXL enhancement:", cfg.run_brainit_sdxl_enhancement)
    return combined


def final_recon_tag(cfg):
    variant = f"s{cfg.diffusion_start_step}_n{cfg.diffusion_num_steps}"
    if cfg.semantic_target == "text_tokens":
        variant = f"sdxltext_c{cfg.caption_index}_g{cfg.sdxl_guidance_scale:g}_" + variant
    if not cfg.use_lowlevel_init_for_diffusion:
        variant += "_no_lowlevel"
    if not cfg.use_clip_tokens_for_diffusion:
        variant += "_no_clip"
    if cfg.diffusion_ensemble_size > 1:
        variant += f"_bestof{cfg.diffusion_ensemble_size}_lowlevel_mse"
    if not cfg.run_brainit_sdxl_enhancement:
        variant += "_no_sdxl"
    if cfg.oracle_clip:
        variant += "_oracle_clip"
    if cfg.diffusion_experiment_name:
        variant += "_" + str(cfg.diffusion_experiment_name).strip().replace(" ", "_").replace("/", "-")
    return f"singlehead_finaleval{cfg.final_eval_images}_{variant}"


def _diffusion_paths(cfg, out_dir, tag):
    s = cfg.subject
    semantic_dir = f"{out_dir}/{s}_brainit_semantic_uint8_{tag}"
    os.makedirs(semantic_dir, exist_ok=True)
    return {
        "semantic_recon_dir": semantic_dir,
        "semantic_recons_path": f"{out_dir}/{s}_brainit_semantic_uint8_{tag}.pt",
        "semantic_ids_path": f"{out_dir}/{s}_brainit_semantic_img_ids_{tag}.json",
        "enhanced_recons_path": f"{out_dir}/{s}_brainit_enhanced_uint8_{tag}.pt",
        "enhanced_ids_path": f"{out_dir}/{s}_brainit_enhanced_img_ids_{tag}.json",
    }


def _part_path(paths, i):
    return f"{paths['semantic_recon_dir']}/img_{i:04d}.pt"


def _ensemble_part_path(paths, i):
    return f"{paths['semantic_recon_dir']}/img_{i:04d}_ensemble.pt"


def _load_ids(path):
    with open(path) as f:
        return [int(i) for i in json.load(f)]


def load_diffusion_inputs(cfg, data, dip_manifest, prediction):
    lowlevel_images = torch_load_cpu(dip_manifest["recons_path"]).float().clamp(0, 1)
    lowlevel_ids = _load_ids(dip_manifest["img_ids_path"])[:len(lowlevel_images)]
    shared_ids = [int(i) for i in prediction["shared_img_ids"]]
    if any(i not in data.final_heldout_ids for i in lowlevel_ids + shared_ids):
        raise ValueError(f"{cfg.subject} diffusion inputs include non-heldout image IDs")
    rows = rows_for_ids(shared_ids, lowlevel_ids, f"{cfg.subject} predicted CLIP tokens")
    clip_tokens = prediction["pred_clip_tokens"][rows].float().contiguous()
    if tuple(clip_tokens.shape[1:]) != (CLIP_NUM_QUERIES, cfg.semantic_dim):
        raise ValueError(f"expected semantic tokens [B, {CLIP_NUM_QUERIES}, {cfg.semantic_dim}], got {tuple(clip_tokens.shape)}")
    if len(clip_tokens) != len(lowlevel_images):
        raise ValueError(f"low-level/CLIP row mismatch: lowlevel={len(lowlevel_images)} clip={len(clip_tokens)}")
    return lowlevel_images.contiguous(), clip_tokens, lowlevel_ids


def select_best_candidates(cfg, indices, candidates_uint8, lowlevel_images):
    """Best-of-N: keep the candidate closest (pixel MSE) to the low-level DIP image. No ground truth used."""
    if cfg.diffusion_ensemble_size == 1:
        return candidates_uint8, None, None
    n = len(indices)
    candidates = candidates_uint8.reshape(n, cfg.diffusion_ensemble_size, *candidates_uint8.shape[1:])
    ref = F.interpolate(lowlevel_images[indices].float().clamp(0, 1), size=candidates.shape[-2:], mode="bicubic", align_corners=False)
    scores = (candidates.float().div(255.0) - ref[:, None]).square().mean(dim=(2, 3, 4))
    best = scores.argmin(dim=1)
    return candidates[torch.arange(n), best].contiguous(), best.cpu(), scores.cpu()


@torch.inference_mode()
def _generate(cfg, combined, indices, clip_tokens_all, lowlevel_images):
    idx = torch.as_tensor(indices, dtype=torch.long)
    clip_tokens = clip_tokens_all[idx].to(DEVICE, non_blocking=True).float()
    if not cfg.use_clip_tokens_for_diffusion:
        clip_tokens = torch.zeros_like(clip_tokens)
    img_init = None
    if cfg.use_lowlevel_init_for_diffusion:
        img_init = lowlevel_images[idx].permute(0, 2, 3, 1).float().clamp(0, 1).cpu().numpy()
    n = int(clip_tokens.shape[0])
    dummy = torch.empty(n, device=DEVICE, dtype=torch.long)
    samples = combined.generate(
        clip_tokens, dummy, dummy,
        img_init=img_init,
        num_samples=n,
        start_step=cfg.diffusion_start_step,
        num_steps=cfg.diffusion_num_steps,
    )
    return samples.float().clamp(0, 1).cpu().mul(255).round().byte()


def _generate_safe(cfg, combined, indices, clip_tokens, lowlevel_images):
    try:
        return _generate(cfg, combined, indices, clip_tokens, lowlevel_images)
    except torch.cuda.OutOfMemoryError:
        if not cfg.diffusion_auto_split_on_oom or len(indices) <= 1:
            raise
    # Retry outside the except block: inside it, the traceback still pins the failed attempt's tensors.
    torch.cuda.empty_cache()
    mid = len(indices) // 2
    return torch.cat([
        _generate_safe(cfg, combined, indices[:mid], clip_tokens, lowlevel_images),
        _generate_safe(cfg, combined, indices[mid:], clip_tokens, lowlevel_images),
    ], dim=0)


def _load_semantic_cache(cfg, paths, ids):
    if cfg.rerun_diffusion or not os.path.exists(paths["semantic_recons_path"]):
        return None
    cached = torch_load_cpu(paths["semantic_recons_path"])
    cached_ids = _load_ids(paths["semantic_ids_path"]) if os.path.exists(paths["semantic_ids_path"]) else None
    if cached_ids != ids or len(cached) != len(ids):
        print("rejecting stale semantic diffusion cache:", paths["semantic_recons_path"])
        return None
    return cached


def diffuse(cfg, combined, data, dip_manifest, prediction, out_dir, on_batch_end=noop):
    tag = final_recon_tag(cfg)
    lowlevel_images, clip_tokens, ids = load_diffusion_inputs(cfg, data, dip_manifest, prediction)
    paths = _diffusion_paths(cfg, out_dir, tag)
    n = len(ids)
    print("diffusion conditioning aligned:", {
        "n": n, "first_ids": ids[:8],
        "clip_shape": tuple(clip_tokens.shape), "lowlevel_shape": tuple(lowlevel_images.shape),
        "clip_mean": float(clip_tokens.mean()), "clip_std": float(clip_tokens.std()),
    })

    semantic = _load_semantic_cache(cfg, paths, ids)
    if semantic is None:
        full_rerun = bool(cfg.rerun_diffusion)
        if os.path.exists(paths["semantic_ids_path"]):
            if _load_ids(paths["semantic_ids_path"]) != ids:
                print("rejecting stale semantic shard cache:", paths["semantic_recon_dir"])
                full_rerun = True
        elif any(os.path.exists(_part_path(paths, i)) for i in range(min(n, 64))):
            print("semantic shards have no image-id sidecar; regenerating shards for current IDs.")
            full_rerun = True
        pending = [i for i in range(n) if full_rerun or not os.path.exists(_part_path(paths, i))]
        k = cfg.diffusion_ensemble_size
        image_batch = max(1, cfg.diffusion_batch_size // k)
        for start in tqdm(range(0, len(pending), image_batch), desc="BrainIT semantic diffusion"):
            batch = pending[start:start + image_batch]
            candidates = _generate_safe(cfg, combined, [i for i in batch for _ in range(k)], clip_tokens, lowlevel_images)
            samples, best, scores = select_best_candidates(cfg, batch, candidates, lowlevel_images)
            for i, sample in zip(batch, samples):
                torch.save(sample, _part_path(paths, i))
            if cfg.save_diffusion_candidates and k > 1:
                by_image = candidates.reshape(len(batch), k, *candidates.shape[1:])
                for row, i in enumerate(batch):
                    torch.save({
                        "img_id": int(ids[i]), "best": int(best[row]),
                        "scores": scores[row], "candidates": by_image[row],
                    }, _ensemble_part_path(paths, i))
            on_batch_end()
            if cfg.diffusion_empty_cache_every and start % (image_batch * cfg.diffusion_empty_cache_every) == 0:
                torch.cuda.empty_cache()
        semantic = torch.stack([torch_load_cpu(_part_path(paths, i)) for i in range(n)]).contiguous()
        torch.save(semantic, paths["semantic_recons_path"])
        with open(paths["semantic_ids_path"], "w") as f:
            json.dump(ids, f)
    print("semantic reconstructions:", tuple(semantic.shape), semantic.dtype, paths["semantic_recons_path"])

    if cfg.run_brainit_sdxl_enhancement:
        if os.path.exists(paths["enhanced_recons_path"]) and not cfg.rerun_diffusion:
            final = torch_load_cpu(paths["enhanced_recons_path"])
        else:
            from utils.diffusion_utils import enhance_recons
            bp = brainit_paths(cfg)
            enhanced_hwc = enhance_recons(
                semantic.float().div(255.0).permute(0, 2, 3, 1).numpy(),
                unclip_config_path=bp["unclip_config"],
                sdxl_config_path=bp["sdxl_config"],
                sdxl_ckpt_path=bp["sdxl_ckpt"],
                num_samples=1,
            )
            enhanced = torch.from_numpy(enhanced_hwc).permute(0, 3, 1, 2).float().clamp(0, 1)
            final = F.interpolate(enhanced, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1).mul(255).round().byte().contiguous()
            torch.save(final, paths["enhanced_recons_path"])
            with open(paths["enhanced_ids_path"], "w") as f:
                json.dump(ids, f)
        final_recons_path, final_ids_path = paths["enhanced_recons_path"], paths["enhanced_ids_path"]
    else:
        final = semantic
        final_recons_path, final_ids_path = paths["semantic_recons_path"], paths["semantic_ids_path"]
    print("final reconstructions:", tuple(final.shape), final.dtype, final_recons_path)

    manifest = {
        "subject": cfg.subject,
        "n": n,
        "final_recon_tag": tag,
        "final_recons_path": final_recons_path,
        "final_ids_path": final_ids_path,
        "lowlevel_recons_path": dip_manifest["recons_path"],
        "lowlevel_ids_path": dip_manifest["img_ids_path"],
        **paths,
    }
    with open(f"{out_dir}/final_diffusion_manifest_{tag}.json", "w") as f:
        json.dump({cfg.subject: manifest}, f, indent=2)
    return manifest
