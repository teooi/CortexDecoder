import json

import torch
import torch.nn as nn

from ..core.config import CLIP_NUM_QUERIES, CORTEXMAE_TEMPORAL, HEADS, ROI_COVERAGE_FILE, VGG_TOKEN_DIM, VGG_TOTAL_TOKENS


class CrossAttentionBridge(nn.Module):
    def __init__(self, brain_dim, target_dim, num_queries, d_model, n_layers, n_heads, ffn_dim, dropout, num_memory=0):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.brain_proj = nn.Linear(brain_dim, d_model) if brain_dim != d_model else nn.Identity()
        self.brain_norm = nn.LayerNorm(d_model)
        # num_memory > 0: learned positional embedding per brain token, added after the norm.
        self.memory_pos = nn.Parameter(torch.randn(num_memory, d_model) * 0.02) if num_memory else None
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, target_dim)

    def forward(self, brain_tokens, query_indices=None):
        b = brain_tokens.shape[0]
        memory = self.brain_norm(self.brain_proj(brain_tokens))
        if self.memory_pos is not None:
            memory = memory + self.memory_pos
        if query_indices is None:
            queries = self.query_embed.unsqueeze(0).expand(b, -1, -1)
        elif query_indices.ndim == 1:
            queries = self.query_embed[query_indices].unsqueeze(0).expand(b, -1, -1)
        else:
            queries = self.query_embed[query_indices]
        out = self.decoder(tgt=queries, memory=memory)
        return self.output_proj(out)


class FlatmapSingleHeadBridge(nn.Module):
    """One bridge per head; state-dict layout matches the v1 notebook checkpoints."""

    def __init__(self, cfg, brain_dim, num_patches, target, roi_patches=None):
        super().__init__()
        if target not in HEADS:
            raise ValueError(f"target must be 'vgg' or 'clip', got {target!r}")
        self.target = target
        self.num_patches = int(num_patches)
        # ROI view (semantic head only): average the 4 time copies of each patch, keep ROI patches, standardize with
        # train-set statistics. Buffers travel with the checkpoint, so prediction applies the same transform.
        if roi_patches is None:
            self.register_buffer("roi_patches", None)
        else:
            self.register_buffer("roi_patches", torch.as_tensor(roi_patches, dtype=torch.long))
            self.register_buffer("roi_mean", torch.zeros(len(roi_patches), brain_dim))
            self.register_buffer("roi_std", torch.ones(len(roi_patches), brain_dim))
            # Standardizing removes each patch's fixed signature, which was the bridge's only cue to patch identity
            # (memory has no positional embedding); a learned per-patch embedding restores it.
            self.roi_pos = nn.Parameter(torch.randn(len(roi_patches), brain_dim) * 0.02)
        if target == "vgg":
            target_dim, num_queries = VGG_TOKEN_DIM, VGG_TOTAL_TOKENS
        else:
            target_dim, num_queries = cfg.semantic_dim, CLIP_NUM_QUERIES
        self.bridge = CrossAttentionBridge(
            brain_dim=brain_dim,
            target_dim=target_dim,
            num_queries=num_queries,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            num_memory=self.num_patches if cfg.memory_pos_embed and roi_patches is None else 0,
        )

    def roi_view(self, brain_tokens):
        """[B, 4 * 364, D] time-major CortexMAE tokens -> [B, n_roi, D]; without standardization if stats are unset."""
        b, n, d = brain_tokens.shape
        x = brain_tokens.reshape(b, CORTEXMAE_TEMPORAL, n // CORTEXMAE_TEMPORAL, d).mean(1)[:, self.roi_patches]
        return x

    def forward(self, brain_tokens, query_indices=None):
        if self.roi_patches is not None:
            brain_tokens = (self.roi_view(brain_tokens) - self.roi_mean) / self.roi_std + self.roi_pos
        return self.bridge(brain_tokens, query_indices=query_indices)


def roi_patches(cfg):
    with open(f"{cfg.data_dir}/{ROI_COVERAGE_FILE}") as f:
        coverage = json.load(f)["coverage"]
    return [p for p, c in enumerate(coverage) if c >= cfg.roi_min_coverage]


def build_bridge(cfg, data, target, device):
    roi = roi_patches(cfg) if target == "clip" and cfg.roi != "all" else None
    model = FlatmapSingleHeadBridge(cfg, data.token_dim, data.num_patches, target, roi_patches=roi).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    shape = f"VGG [{VGG_TOTAL_TOKENS}, {VGG_TOKEN_DIM}]" if target == "vgg" else f"{cfg.semantic_target} [{CLIP_NUM_QUERIES}, {cfg.semantic_dim}]"
    memory = f"roi={cfg.roi} ({len(roi)} patches)" if roi else f"patches={data.num_patches}"
    print(f"initialized {target}: brain_dim={data.token_dim} {memory} target={shape} params={params_m:.1f}M")
    return model


@torch.no_grad()
def fit_roi_stats(model, data, device):
    """Per-patch, per-feature mean/std of the ROI view over training trials (one pass over the sessions)."""
    total = sq = None
    count = 0
    for sess in data.sessions:
        loaded = data.load_session(sess, split="train")
        if loaded is None:
            continue
        for b0 in range(0, len(loaded[0]), 256):
            x = model.roi_view(loaded[0][b0:b0 + 256].to(device).float()).double()
            total = x.sum(0) if total is None else total + x.sum(0)
            sq = (x ** 2).sum(0) if sq is None else sq + (x ** 2).sum(0)
            count += len(x)
    mean = total / count
    model.roi_mean.copy_(mean.float())
    model.roi_std.copy_((sq / count - mean ** 2).clamp_min(1e-12).sqrt().float().clamp_min(1e-4))
    print(f"ROI standardization fitted on {count} training trials")
