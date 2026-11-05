"""The three v1 notebooks as stage functions: train VGG, train CLIP, final diffusion + eval."""
import os

from .core.common import DEVICE, noop, seed_everything
from .data.nsd import NSDSubjectData, shared_ids_path, shared_tokens_path
from .data.stimuli import StimulusImages
from .models.vgg_tokenizer import BrainITVGGTokenizer
from .reconstruction.predict import best_or_latest_checkpoint, load_predictions, predict_shared1000, pred_tokens_path, write_manifest


def _setup(cfg, stage):
    seed_everything(cfg.seed)
    out_dir = cfg.run_dir(stage)
    os.makedirs(out_dir, exist_ok=True)
    print(f"device={DEVICE} stage={stage} batch_size={cfg.batch_size} epochs={cfg.epochs} output={out_dir}")
    return NSDSubjectData(cfg), out_dir


def _manifest_entry(out_dir, subject, heads):
    entry = {f"pred_{h}_path": pred_tokens_path(out_dir, subject, h) for h in heads}
    entry["shared_tokens_path"] = shared_tokens_path(out_dir, subject)
    entry["shared_ids_path"] = shared_ids_path(out_dir, subject)
    return entry


def run_vgg_stage(cfg, commit=noop):
    """production-vgg-train: train VGG head, predict shared-1000 VGG tokens, DIP preview on a subset."""
    from .reconstruction.dip import eval_dip, run_dip_inversion
    from .training.trainer import HeadTrainer

    data, out_dir = _setup(cfg, "vgg")
    stimuli = StimulusImages(cfg.data_dir)
    tokenizer = BrainITVGGTokenizer().to(DEVICE).eval()
    HeadTrainer(cfg, data, "vgg", out_dir, stimuli=stimuli, tokenizer=tokenizer).fit(on_epoch_end=commit)

    predict_shared1000(cfg, data, "vgg", best_or_latest_checkpoint(out_dir, "vgg"), out_dir)
    write_manifest(f"{out_dir}/transfer_vgg_prediction_manifest.json", {cfg.subject: _manifest_entry(out_dir, cfg.subject, ["vgg"])})
    commit()

    prediction = load_predictions(cfg.subject, out_dir, {"vgg": out_dir})
    dip_manifest = run_dip_inversion(cfg, tokenizer, stimuli, prediction, out_dir, on_batch_end=commit)
    eval_dip(cfg, stimuli, dip_manifest, out_dir)
    commit()
    return out_dir


def run_clip_stage(cfg, commit=noop):
    """production-clip-train: train CLIP head, predict shared-1000 CLIP tokens, embedding-level eval."""
    from .evaluation.metrics import clip_embedding_eval
    from .training.trainer import HeadTrainer

    data, out_dir = _setup(cfg, "clip")
    stimuli = StimulusImages(cfg.data_dir) if cfg.decoder_loss_weight > 0 else None
    HeadTrainer(cfg, data, "clip", out_dir, stimuli=stimuli).fit(on_epoch_end=commit)

    predict_shared1000(cfg, data, "clip", best_or_latest_checkpoint(out_dir, "clip"), out_dir)
    write_manifest(f"{out_dir}/transfer_clip_prediction_manifest.json", {cfg.subject: _manifest_entry(out_dir, cfg.subject, ["clip"])})
    clip_embedding_eval(cfg, data, load_predictions(cfg.subject, out_dir, {"clip": out_dir}), out_dir)
    commit()
    return out_dir


def run_final_stage(cfg, commit=noop):
    """production-final-diffusion: best weights -> shared-1000 tokens -> DIP -> CLIP-conditioned diffusion -> eval.

    The low-level side (VGG weights, VGG predictions, DIP images) always comes from `cfg.baseline()`, so a
    semantic-branch ablation reuses the baseline's DIP images and only re-runs CLIP prediction, diffusion and eval.
    For a baseline config both sides share one output dir, as in v1.
    """
    from .reconstruction.diffusion import diffuse, load_combined_model
    from .reconstruction.dip import eval_dip, run_dip_inversion
    from .evaluation.metrics import evaluate_final

    base = cfg.baseline()
    best = {"vgg": f"{base.run_dir('vgg')}/vgg_best.pt", "clip": f"{cfg.run_dir('clip')}/clip_best.pt"}
    for head, path in best.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {head} best weights: {path}")
    data, out_dir = _setup(cfg, "final_diffusion")
    lowlevel_dir = base.run_dir("final_diffusion")
    os.makedirs(lowlevel_dir, exist_ok=True)
    print("source VGG run:", base.run_dir("vgg"))
    print("source CLIP run:", cfg.run_dir("clip"))
    print("low-level (VGG preds + DIP) dir:", lowlevel_dir)

    # A semantic bridge on another brain-token set (e.g. encoder-level ROI) must not reuse the baseline's cached
    # whole-cortex shared-1000 tokens; the VGG side always reads the baseline tokens.
    same_tokens = cfg.brain_tokens == base.brain_tokens
    clip_shared_dir = lowlevel_dir if same_tokens else out_dir
    predict_shared1000(base, data if same_tokens else NSDSubjectData(base), "vgg", best["vgg"], lowlevel_dir)
    predict_shared1000(cfg, data, "clip", best["clip"], out_dir, shared_dir=clip_shared_dir)
    write_manifest(f"{out_dir}/final_diffusion_prediction_manifest.json", {cfg.subject: {
        "pred_vgg_path": pred_tokens_path(lowlevel_dir, cfg.subject, "vgg"),
        "pred_clip_path": pred_tokens_path(out_dir, cfg.subject, "clip"),
        "shared_tokens_path": shared_tokens_path(clip_shared_dir, cfg.subject),
        "shared_ids_path": shared_ids_path(lowlevel_dir, cfg.subject),
        "source_vgg_best_weights_path": best["vgg"],
        "source_clip_best_weights_path": best["clip"],
    }})
    commit()

    prediction = load_predictions(cfg.subject, lowlevel_dir, {"vgg": lowlevel_dir, "clip": out_dir})
    if cfg.oracle_clip and cfg.semantic_target != "text_tokens":
        print(f"ORACLE: conditioning diffusion on ground-truth {cfg.semantic_target}")
        prediction["pred_clip_tokens"] = data.clip_targets_for_ids(prediction["shared_img_ids"]).half()
    stimuli = StimulusImages(cfg.data_dir)
    tokenizer = BrainITVGGTokenizer().to(DEVICE).eval()
    dip_manifest = run_dip_inversion(base, tokenizer, stimuli, prediction, lowlevel_dir, on_batch_end=commit)
    eval_dip(base, stimuli, dip_manifest, lowlevel_dir)
    del tokenizer
    commit()

    if cfg.semantic_target == "text_tokens":
        if not cfg.oracle_clip:
            raise NotImplementedError("text-token bridge not built yet; run text_tokens with --oracle-clip")
        from .reconstruction.sdxl_text import diffuse_text, load_sdxl_img2img
        print("ORACLE: conditioning SDXL on ground-truth caption embeddings")
        pipe = load_sdxl_img2img(cfg)
        diffusion_manifest = diffuse_text(cfg, pipe, data, dip_manifest, out_dir, on_batch_end=commit)
        del pipe
    else:
        if cfg.semantic_target == "siglip2_tokens":
            from .reconstruction.siglip2_unclip import SigLIP2UnCLIP
            combined = SigLIP2UnCLIP(cfg)
        else:
            combined = load_combined_model(cfg)
        diffusion_manifest = diffuse(cfg, combined, data, dip_manifest, prediction, out_dir, on_batch_end=commit)
        combined.to("cpu")
        del combined
    commit()

    evaluate_final(cfg, data, stimuli, diffusion_manifest, prediction, out_dir)
    commit()
    return out_dir
