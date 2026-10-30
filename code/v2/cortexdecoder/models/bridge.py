import torch
import torch.nn as nn

from ..core.config import CLIP_DIM, CLIP_NUM_QUERIES, HEADS, VGG_TOKEN_DIM, VGG_TOTAL_TOKENS


class CrossAttentionBridge(nn.Module):
    def __init__(self, brain_dim, target_dim, num_queries, d_model, n_layers, n_heads, ffn_dim, dropout):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.brain_proj = nn.Linear(brain_dim, d_model) if brain_dim != d_model else nn.Identity()
        self.brain_norm = nn.LayerNorm(d_model)
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

    def __init__(self, cfg, brain_dim, num_patches, target):
        super().__init__()
        if target not in HEADS:
            raise ValueError(f"target must be 'vgg' or 'clip', got {target!r}")
        self.target = target
        self.num_patches = int(num_patches)
        if target == "vgg":
            target_dim, num_queries = VGG_TOKEN_DIM, VGG_TOTAL_TOKENS
        else:
            target_dim, num_queries = CLIP_DIM, CLIP_NUM_QUERIES
        self.bridge = CrossAttentionBridge(
            brain_dim=brain_dim,
            target_dim=target_dim,
            num_queries=num_queries,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
        )

    def forward(self, brain_tokens, query_indices=None):
        return self.bridge(brain_tokens, query_indices=query_indices)


def build_bridge(cfg, data, target, device):
    model = FlatmapSingleHeadBridge(cfg, data.token_dim, data.num_patches, target).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    shape = f"VGG [{VGG_TOTAL_TOKENS}, {VGG_TOKEN_DIM}]" if target == "vgg" else f"CLIP [{CLIP_NUM_QUERIES}, {CLIP_DIM}]"
    print(f"initialized {target}: brain_dim={data.token_dim} patches={data.num_patches} target={shape} params={params_m:.1f}M")
    return model
