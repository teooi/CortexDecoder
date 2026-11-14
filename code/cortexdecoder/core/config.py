"""Run configuration. Defaults are the whole-cortex baseline; the README lists the overrides for the paper's final model."""
import re
import time
from dataclasses import dataclass, fields, replace

import numpy as np

DATA_DIR = "/mnt/braid2-flatmap-data"

# BrainIT low-level branch settings from paper Appendix D.1/D.2.
VGG_IMAGE_SIZE = 112
VGG_TOKEN_DIM = 512
VGG_LAYER_NAMES = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
VGG_LAYER_TOKEN_COUNTS = [56 * 56, 55 * 55, 28 * 28, 14 * 14, 7 * 7]
VGG_LAYER_OFFSETS = np.cumsum([0] + VGG_LAYER_TOKEN_COUNTS[:-1]).tolist()
VGG_TOTAL_TOKENS = int(sum(VGG_LAYER_TOKEN_COUNTS))
SAMPLE_TOKENS_PER_LAYER = [512, 512, 128, 64, 16]

# OpenCLIP ViT-bigG/14 internal image patch tokens.
CLIP_DIM = 1664
CLIP_NUM_QUERIES = 256

SHARED1000_EVAL_SIZE = 1000
TRIALS_PER_SESSION = 750
# Both strings are part of every run-directory name, so they stay fixed.
SPLIT_POLICY = "shared1000_image_id_holdout"
ARCH_MODE = "baseline"
HEADS = ("vgg", "clip")

# Ablation knobs -> short run-dir tag. A run whose knobs are all at their defaults gets no tag,
# so baseline run dirs keep their original names.
ABLATION_TAGS = {"clip_loss": "loss", "clip_temperature": "T", "n_layers": "L", "n_heads": "H",
                 "decoder_loss_weight": "dec", "brain_tokens": "tok", "vgg_brain_tokens": "vtok"}


@dataclass(frozen=True)
class Config:
    subject: str = "subj01"
    timestamp: str = "20260907_013858"
    vgg_timestamp: str = ""  # final stage: VGG run to pair with; defaults to `timestamp`
    seed: int = 42
    final_eval_images: int = 48  # 1000 for the full shared-1000 eval set
    data_dir: str = DATA_DIR
    known_bad_sessions: tuple = (("subj01", 11),)
    val_frac: float = 0.05

    # Cross-attention bridge.
    d_model: int = 768
    n_layers: int = 4
    n_heads: int = 8
    ffn_dim: int = 3072
    dropout: float = 0.10

    # Training.
    batch_size: int = 128
    epochs: int = 30
    lr: float = 5e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 2  # linear LR warmup
    plateau_patience: int = 5  # a patience of 2 cut the LR at the first noisy epoch after warmup
    plateau_factor: float = 0.1
    grad_clip_norm: float = 1.0
    temperature: float = 0.07  # VGG InfoNCE
    clip_loss: str = "mse"  # see training/losses.py:build_clip_loss, e.g. "mse+0.1*infonce"
    clip_temperature: float = 0.07  # CLIP InfoNCE / SoftCLIP terms
    best_metric: str = "val_loss"  # "val_loss" | "val_cos"; use val_cos to compare runs with different losses
    # Decoder-aware fine-tuning: clip_loss + decoder_loss_weight * frozen unCLIP diffusion loss (training/decoder_loss.py).
    decoder_loss_weight: float = 0.0
    decoder_loss_batch: int = 32  # samples per batch that get the (costly) diffusion term; the clip loss uses all
    decoder_t_max: int = 640  # highest timestep visited by img2img from start step 14 of 38
    init_checkpoint: str = ""  # bridge weights to start from (fine-tuning) when the run has no checkpoint yet
    # Brain-token set under data_dir: "flatmap_tokens_fp16" (whole cortex) or "flatmap_tokens_roi-nsdgeneral_fp16"
    # (visual-cortex ROI, scripts/generate_roi_tokens.py: only nsdgeneral patches visible to CortexMAE).
    brain_tokens: str = "flatmap_tokens_fp16"
    # Final stage: brain-token set of the paired VGG run (see baseline()); "" pairs with the whole-cortex VGG bridge.
    vgg_brain_tokens: str = ""
    cache_sessions_in_ram: bool = False

    # Shared-1000 prediction.
    predict_batch: int = 1
    predict_query_chunk: int = 1024

    # Deep Image Prior inversion.
    dip_input_dim: int = 32
    dip_internal_dim: int = 128
    dip_steps: int = 2000
    dip_lr: float = 1e-2
    dip_initial_noise: float = 0.1
    dip_reg_noise: float = 1.0 / 30.0
    dip_ema: float = 0.99
    dip_output_size: int = 112
    dip_upsampled_size: int = 256
    dip_batch_size: int = 36
    dip_min_batch_size: int = 1
    dip_layer_weights: tuple = (1.0, 1.0, 1.0, 1.0, 1.0)
    rerun_dip: bool = False

    # BrainIT/MindEye2 diffusion.
    diffusion_start_step: int = 14
    diffusion_num_steps: int = 38
    diffusion_ensemble_size: int = 1  # 1 = one sample per image (main results); N > 1 keeps the candidate closest to the DIP image
    diffusion_batch_size: int = 40  # total candidates per model call
    save_diffusion_candidates: bool = True
    diffusion_auto_split_on_oom: bool = True
    diffusion_empty_cache_every: int = 20
    rerun_diffusion: bool = False

    # Final evaluation.
    eval_feature_batch_size: int = 32
    run_swav_eval: bool = True
    preview_grid_n: int = 48
    preview_grid_cols: int = 6

    def __post_init__(self):
        n = min(int(self.final_eval_images), SHARED1000_EVAL_SIZE)
        if n < 1:
            raise ValueError("final_eval_images must be at least 1")
        object.__setattr__(self, "final_eval_images", n)
        if not str(self.timestamp).strip():
            object.__setattr__(self, "timestamp", time.strftime("%Y%m%d_%H%M%S"))

    @property
    def subject_tag(self):
        return "subject" + self.subject[-2:]

    @property
    def output_root(self):
        return f"{self.data_dir}/outputs"

    @property
    def ablation_tag(self):
        defaults = {f.name: f.default for f in fields(Config)}
        parts = [f"{short}-{getattr(self, name)}" for name, short in ABLATION_TAGS.items() if getattr(self, name) != defaults[name]]
        return re.sub(r"[^A-Za-z0-9.\-]+", "_", "_".join(parts))

    def run_dir(self, stage):
        """stage: 'vgg', 'clip' or 'final_diffusion'."""
        name = (
            f"production_{stage}_{self.subject_tag}_baseline_"
            f"{SPLIT_POLICY}_{ARCH_MODE}_subj{self.subject[-2:]}_{self.timestamp}"
        )
        return f"{self.output_root}/{name}" + (f"_{self.ablation_tag}" if self.ablation_tag else "")

    def baseline(self):
        """The VGG run (and its DIP images) this run pairs with: same run with ablation knobs reset.

        Only `brain_tokens` survives, set from `vgg_brain_tokens`, so the VGG side can be a visual-cortex ROI run too.
        """
        defaults = {f.name: f.default for f in fields(Config)}
        reset = {name: defaults[name] for name in ABLATION_TAGS}
        reset["brain_tokens"] = self.vgg_brain_tokens or defaults["brain_tokens"]
        return replace(self, timestamp=self.vgg_timestamp or self.timestamp, vgg_timestamp="", **reset)

    def clip_source(self):
        """Config whose run_dir('clip') holds this run's semantic bridge; `vgg_brain_tokens` only concerns the final stage."""
        return replace(self, vgg_brain_tokens="")

    def with_overrides(self, **overrides):
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


def vgg_config(**overrides):
    return Config(batch_size=128, epochs=30).with_overrides(**overrides)


def clip_config(**overrides):
    return Config(batch_size=64, epochs=25).with_overrides(**overrides)


def final_config(**overrides):
    return Config(final_eval_images=1000).with_overrides(**overrides)
