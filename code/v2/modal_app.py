"""CortexDecoder v2 on Modal (subject 1).

    modal run code/v2/modal_app.py                          # all stages, default timestamp (reuses existing runs)
    modal run code/v2/modal_app.py --stage vgg --timestamp new
    modal run code/v2/modal_app.py --stage final --final-eval-images 48
    modal run --detach code/v2/modal_app.py --stage all --timestamp new

Stages mirror the v1 notebooks: `vgg` (production-vgg-train), `clip` (production-clip-train),
`final` (production-final-diffusion). `all` trains VGG and CLIP in parallel, then runs `final`.
Every stage resumes from what is already on the volume for the given timestamp.
"""
import time
from pathlib import Path

import modal

VOLUME_NAME = "braid2-flatmap-data"
DATA_DIR = "/mnt/braid2-flatmap-data"
DEFAULT_TIMESTAMP = "20260907_013858"
GPU = "H100"
TIMEOUT_S = 24 * 60 * 60

volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "xformers==0.0.32.post2",
        index_url="https://download.pytorch.org/whl/cu129",
    )
    .pip_install(
        "h5py",
        "numpy",
        "pandas",
        "scipy",
        "scikit-image",
        "tqdm",
        "matplotlib",
        # BrainIT / MindEye2 diffusion stack.
        "open_clip_torch==2.24.0",
        "clip-anytorch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "omegaconf",
        "pytorch-lightning",
        "einops",
        "kornia",
        "timm",
        "torch-geometric",
        "webdataset",
        "fsspec",
        "requests",
        "huggingface_hub",
        "setuptools<81",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(Path(__file__).parent / "cortexdecoder", "/root/cortexdecoder")
)

app = modal.App("cortexdecoder-v2", image=image)

gpu_function = app.function(
    gpu=GPU,
    volumes={DATA_DIR: volume},
    timeout=TIMEOUT_S,
    memory=65536,
)


def _throttled_commit(every_s=300):
    """Persist volume writes at most every `every_s` seconds so long runs survive preemption."""
    last = [time.monotonic()]

    def commit(force=False):
        if force or time.monotonic() - last[0] >= every_s:
            volume.commit()
            last[0] = time.monotonic()

    return commit


@gpu_function
def train_vgg(overrides: dict):
    from cortexdecoder.core.config import vgg_config
    from cortexdecoder.pipeline import run_vgg_stage

    commit = _throttled_commit()
    out_dir = run_vgg_stage(vgg_config(**overrides), commit=commit)
    commit(force=True)
    return out_dir


@gpu_function
def train_clip(overrides: dict):
    from cortexdecoder.core.config import clip_config
    from cortexdecoder.pipeline import run_clip_stage

    commit = _throttled_commit()
    out_dir = run_clip_stage(clip_config(**overrides), commit=commit)
    commit(force=True)
    return out_dir


@gpu_function
def final_diffusion(overrides: dict):
    from cortexdecoder.core.config import final_config
    from cortexdecoder.pipeline import run_final_stage

    volume.reload()
    commit = _throttled_commit()
    out_dir = run_final_stage(final_config(**overrides), commit=commit)
    commit(force=True)
    return out_dir


@gpu_function
def decoder_loss_probe(overrides: dict):
    """One training batch through the decoder-aware loss: memory, step time, gradient sizes, and whether the frozen
    unCLIP's diffusion loss actually depends on the conditioning (true < predicted < shuffled tokens)."""
    import torch

    from cortexdecoder.core.common import DEVICE
    from cortexdecoder.core.config import clip_config
    from cortexdecoder.data.nsd import NSDSubjectData
    from cortexdecoder.data.stimuli import StimulusImages
    from cortexdecoder.training.trainer import HeadTrainer

    cfg = clip_config(**overrides)
    data = NSDSubjectData(cfg)
    trainer = HeadTrainer(cfg, data, "clip", "/tmp/probe", stimuli=StimulusImages(cfg.data_dir))
    trainer._resume()  # loads init_checkpoint
    volume.commit()  # persist the VAE latent cache
    brain, ids = data.load_session(data.sessions[0], split="train")
    brain, ids = brain[:cfg.batch_size].to(DEVICE).float(), ids[:cfg.batch_size]
    k = cfg.decoder_loss_batch
    target = data.clip_targets_for_ids(ids).to(DEVICE).float()
    loss_fn, model = trainer.decoder_loss, trainer.model

    def decoder_loss(tokens, seed=0, ids_=ids):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return float(loss_fn(tokens[:k], ids_[:k], generator=gen))

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(brain)
    seeds = range(8)  # average over noise draws: the per-batch diffusion loss is noisy
    report = {name: sum(decoder_loss(tok, s) for s in seeds) / len(seeds) for name, tok in [
        ("true_tokens", target), ("predicted_tokens", pred), ("shuffled_true_tokens", target.roll(1, dims=0)),
        ("zero_tokens", torch.zeros_like(target))]}

    grads = {}
    for name in ["clip", "decoder"]:
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(brain)
            loss = trainer.clip_loss_fn(out, target)[0] if name == "clip" else loss_fn(out[:k], ids[:k])
        loss.backward()
        torch.cuda.synchronize()
        grads[name] = {"loss": float(loss), "seconds": time.monotonic() - t0,
                       "grad_norm": float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))),
                       "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    report["grads"] = grads
    report["lambda_for_equal_grad_norm"] = grads["clip"]["grad_norm"] / max(grads["decoder"]["grad_norm"], 1e-12)
    print(report)
    return report


@app.function(volumes={DATA_DIR: volume}, timeout=30 * 60, memory=32768)
def smoke_test():
    """CPU-only checks against real volume data: splits, both bridge heads, losses, VGG token round-trip, figures."""
    import tempfile

    import torch

    from cortexdecoder.core.config import VGG_TOTAL_TOKENS, vgg_config
    from cortexdecoder.data.nsd import NSDSubjectData
    from cortexdecoder.data.stimuli import StimulusImages
    from cortexdecoder.reconstruction.dip import tokens_to_brainit_vgg_features
    from cortexdecoder.core.figures import plt, save_figure
    from cortexdecoder.training.losses import sample_layer_indices, select_target_tokens, vgg_contrastive_loss
    from cortexdecoder.models.bridge import build_bridge
    from cortexdecoder.models.vgg_tokenizer import BrainITVGGTokenizer

    cfg = vgg_config()
    data = NSDSubjectData(cfg)
    brain, ids = data.load_session(data.sessions[0], split="train")
    brain, ids = brain[:4].float(), ids[:4]
    images = StimulusImages(cfg.data_dir).for_ids(ids)
    tokenizer = BrainITVGGTokenizer().eval()
    with torch.no_grad():
        target_layers = tokenizer(images)
        feats = tokenizer.extract_features(images)
    query_ids, splits = sample_layer_indices("cpu", train=True)
    target = select_target_tokens(target_layers, query_ids, splits)
    vgg = build_bridge(cfg, data, "vgg", "cpu")
    loss, _ = vgg_contrastive_loss(vgg(brain, query_indices=query_ids), target, splits, cfg.temperature)
    print("vgg loss:", float(loss))
    clip = build_bridge(cfg, data, "clip", "cpu")
    print("clip pred:", tuple(clip(brain).shape))

    all_tokens = torch.cat(target_layers, dim=1)
    assert all_tokens.shape[1] == VGG_TOTAL_TOKENS, all_tokens.shape
    for i, (rebuilt, ref) in enumerate(zip(tokens_to_brainit_vgg_features(all_tokens), feats)):
        err = float((rebuilt - ref).abs().max())
        print(f"vgg layer {i} token round-trip max abs err: {err:.2e}")
        assert err < 1e-4, f"layer {i} round-trip mismatch"

    fig = plt.figure()
    save_figure(fig, f"{tempfile.mkdtemp()}/dpi_check.png")
    try:
        save_figure(plt.figure(), "/tmp/bad.png", dpi=180)
        raise AssertionError("dpi guard did not trigger")
    except ValueError:
        pass

    _check_semantic_losses(data, brain, ids)
    print("smoke test passed")


def _check_semantic_losses(data, brain, ids):
    """Every registered semantic loss (and a weighted combo) must give finite gradients through the CLIP bridge
    and decrease when overfitting a 4-image batch of real CLIP targets. Also checks ablation run naming/guards."""
    import torch

    from cortexdecoder.core.config import clip_config
    from cortexdecoder.models.bridge import build_bridge
    from cortexdecoder.training.losses import CLIP_LOSSES, build_clip_loss
    from cortexdecoder.training.trainer import architecture_name, assert_checkpoint_compatible

    target = data.clip_targets_for_ids(ids).float()
    for name in [*CLIP_LOSSES, "mse+0.5*cosine"]:
        cfg = clip_config(clip_loss=name, n_layers=1)
        torch.manual_seed(0)
        model = build_bridge(cfg, data, "clip", "cpu")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = build_clip_loss(cfg)
        losses = []
        for _ in range(15):
            loss, parts = loss_fn(model(brain), target)
            opt.zero_grad()
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), name
            opt.step()
            losses.append(float(loss))
        print(f"clip loss {name:16s} {losses[0]:.4f} -> {losses[-1]:.4f} parts={sorted(parts)}")
        assert losses[-1] < losses[0], f"{name} did not decrease"

    try:
        build_clip_loss(clip_config(clip_loss="mse+huber"))
        raise AssertionError("unknown loss accepted")
    except ValueError:
        pass

    base, abl = clip_config(), clip_config(clip_loss="mse+0.5*cosine", timestamp="x", vgg_timestamp="20260907_013858")
    assert base.ablation_tag == "" and base.run_dir("clip").endswith("_20260907_013858"), base.run_dir("clip")
    assert abl.run_dir("clip").endswith("_x_loss-mse_0.5_cosine"), abl.run_dir("clip")
    assert abl.baseline().run_dir("vgg") == base.run_dir("vgg"), abl.baseline().run_dir("vgg")
    v1_ckpt = {"split_policy": "shared1000_image_id_holdout", "val_frac": 0.05, "domain": "clip",
               "architecture": architecture_name(base, "clip"), "n_layers": 4}
    assert_checkpoint_compatible(v1_ckpt, "v1", base, "clip")
    try:
        assert_checkpoint_compatible(v1_ckpt, "v1", abl, "clip")
        raise AssertionError("resumed an MSE checkpoint under a different clip_loss")
    except ValueError:
        pass
    print("ablation naming + checkpoint guards ok")

    import tempfile

    from cortexdecoder.evaluation.metrics import clip_embedding_eval

    shared = sorted(data.shared_ids)[:50]
    true = data.clip_targets_for_ids(shared).float()
    # Exact predictions must be perfectly identifiable; row-shuffled predictions must sit at chance (~0.5).
    for name, pred in [("exact", true), ("shuffled", true[torch.randperm(len(true))])]:
        fake = {"shared_img_ids": shared, "pred_clip_tokens": pred.half()}
        row = clip_embedding_eval(clip_config(final_eval_images=50), data, fake, tempfile.mkdtemp())
        print(f"{name}: two_way={row['token_two_way']:.3f} retrieval={row['token_retrieval_img']:.3f}")
        assert (row["token_two_way"] > 0.99) if name == "exact" else (0.3 < row["token_two_way"] < 0.7), row
    print("clip embedding proxies ok")


ABLATION_KEYS = ("clip_loss", "clip_temperature", "n_layers", "n_heads")


def _parse_extra(extra):
    """--extra "batch_size=256,lr=1e-3,best_metric=val_cos" -> any Config field; values are typed by literal."""
    out = {}
    for item in filter(None, (s.strip() for s in extra.split(","))):
        key, _, raw = item.partition("=")
        for cast in (int, float):
            try:
                out[key.strip()] = cast(raw)
                break
            except ValueError:
                continue
        else:
            out[key.strip()] = {"true": True, "false": False}.get(raw.lower(), raw)
    return out


@app.local_entrypoint()
def main(
    stage: str = "all",
    timestamp: str = DEFAULT_TIMESTAMP,
    vgg_timestamp: str = "",
    epochs: int = 0,
    final_eval_images: int = 0,
    clip_loss: str = "",
    clip_temperature: float = 0.0,
    n_layers: int = 0,
    n_heads: int = 0,
    oracle_clip: bool = False,
    semantic_target: str = "",
    extra: str = "",
):

    """--timestamp new starts fresh runs; 0 / "" keep each stage's notebook default.

    Semantic-loss ablation against an existing VGG run:
        modal run code/v2/modal_app.py --stage clip  --timestamp new --clip-loss "mse+0.5*cosine"
        modal run code/v2/modal_app.py --stage final --timestamp <that ts> --clip-loss "mse+0.5*cosine" --vgg-timestamp 20260907_013858
    """
    if timestamp == "new":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    overrides = {"timestamp": timestamp}
    for key, value in [
        ("vgg_timestamp", vgg_timestamp), ("epochs", epochs), ("final_eval_images", final_eval_images),
        ("clip_loss", clip_loss), ("clip_temperature", clip_temperature), ("n_layers", n_layers), ("n_heads", n_heads),
        ("oracle_clip", oracle_clip), ("semantic_target", semantic_target),
    ]:
        if value:
            overrides[key] = value
    overrides.update(_parse_extra(extra))
    print(f"stage={stage} overrides={overrides}")

    if stage == "vgg":
        print(train_vgg.remote(overrides))
    elif stage == "clip":
        print(train_clip.remote(overrides))
    elif stage == "final":
        print(final_diffusion.remote(overrides))
    elif stage == "all":
        # The final stage always pairs with the baseline VGG run, so ablation knobs only go to CLIP.
        vgg_call = train_vgg.spawn({k: v for k, v in overrides.items() if k not in ABLATION_KEYS})
        clip_call = train_clip.spawn(overrides)
        print(vgg_call.get())
        print(clip_call.get())
        print(final_diffusion.remote(overrides))
    else:
        raise ValueError(f"unknown stage {stage!r}; use vgg, clip, final or all")
