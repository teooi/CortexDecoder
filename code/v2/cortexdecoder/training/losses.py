import torch
import torch.nn.functional as F

from ..core.config import SAMPLE_TOKENS_PER_LAYER, VGG_LAYER_NAMES, VGG_LAYER_OFFSETS, VGG_LAYER_TOKEN_COUNTS


def sample_layer_indices(device, train=True):
    """Pick K query positions per VGG layer: random when training, evenly spaced otherwise."""
    query_ids, splits = [], []
    for offset, n_tokens, k in zip(VGG_LAYER_OFFSETS, VGG_LAYER_TOKEN_COUNTS, SAMPLE_TOKENS_PER_LAYER):
        k = min(k, n_tokens)
        if train:
            ids = torch.randperm(n_tokens, device=device)[:k]
        else:
            ids = torch.linspace(0, n_tokens - 1, steps=k, device=device).long()
        query_ids.append(ids + offset)
        splits.append(k)
    return torch.cat(query_ids), splits


def select_target_tokens(tokens_by_layer, query_ids, splits):
    chunks, start = [], 0
    for li, (offset, k) in enumerate(zip(VGG_LAYER_OFFSETS, splits)):
        chunks.append(tokens_by_layer[li][:, query_ids[start:start + k] - offset])
        start += k
    return torch.cat(chunks, dim=1)


def split_layers(x, splits):
    out, start = [], 0
    for k in splits:
        out.append(x[:, start:start + k])
        start += k
    return out


def info_nce_by_position(pred, target, temperature):
    b, k, _ = pred.shape
    if b < 2:
        return F.mse_loss(pred, target)
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    logits = torch.einsum("bkd,ckd->kbc", pred, target) / temperature
    labels = torch.arange(b, device=pred.device).repeat(k)
    loss_i = F.cross_entropy(logits.reshape(k * b, b), labels)
    loss_t = F.cross_entropy(logits.transpose(1, 2).reshape(k * b, b), labels)
    return 0.5 * (loss_i + loss_t)


def vgg_contrastive_loss(pred, target, splits, temperature):
    loss = pred.new_tensor(0.0)
    parts = {}
    for name, p, t in zip(VGG_LAYER_NAMES, split_layers(pred, splits), split_layers(target, splits)):
        layer_loss = info_nce_by_position(p, t, temperature)
        parts[f"nce_{name}"] = layer_loss
        loss = loss + layer_loss
    return loss / len(splits), parts


# Semantic-branch objectives. pred/target: [B, 256, 1664] OpenCLIP tokens.

def clip_mse(pred, target, cfg):
    return F.mse_loss(pred.float(), target)


def clip_cosine(pred, target, cfg):
    """1 - per-token cosine similarity."""
    return 1.0 - F.cosine_similarity(pred.float(), target, dim=-1).mean()


def clip_infonce(pred, target, cfg):
    """Symmetric InfoNCE across the batch on mean-pooled tokens."""
    return info_nce_by_position(pred.float().mean(1, keepdim=True), target.mean(1, keepdim=True), cfg.clip_temperature)


def clip_token_infonce(pred, target, cfg):
    """Symmetric InfoNCE across the batch at every token position (same form as the VGG branch)."""
    return info_nce_by_position(pred, target, cfg.clip_temperature)


def clip_softclip(pred, target, cfg):
    """MindEye's SoftCLIP: InfoNCE on flattened tokens with soft labels from target-target similarity,
    so near-duplicate images in the batch are not treated as hard negatives."""
    p = F.normalize(pred.float().flatten(1), dim=-1)
    t = F.normalize(target.float().flatten(1), dim=-1)
    soft = (t @ t.T / cfg.clip_temperature).softmax(-1)
    logits = p @ t.T / cfg.clip_temperature
    loss_p = -(logits.log_softmax(-1) * soft).sum(-1).mean()
    loss_t = -(logits.T.log_softmax(-1) * soft).sum(-1).mean()
    return 0.5 * (loss_p + loss_t)


CLIP_LOSSES = {
    "mse": clip_mse,
    "cosine": clip_cosine,
    "infonce": clip_infonce,
    "token_infonce": clip_token_infonce,
    "softclip": clip_softclip,
}


def build_clip_loss(cfg):
    """Parse Config.clip_loss, a weighted sum of registry terms, e.g. "mse", "mse+0.5*cosine", "cosine+0.1*infonce"."""
    terms = []
    for part in cfg.clip_loss.replace(" ", "").split("+"):
        weight, _, name = part.rpartition("*")
        if name not in CLIP_LOSSES:
            raise ValueError(f"unknown clip loss term {name!r} in {cfg.clip_loss!r}; options: {sorted(CLIP_LOSSES)}")
        terms.append((float(weight) if weight else 1.0, name, CLIP_LOSSES[name]))

    def loss_fn(pred, target):
        parts = {name: fn(pred, target, cfg) for _, name, fn in terms}
        total = torch.stack([w * parts[name] for w, name, _ in terms]).sum()
        return total, (parts if len(terms) > 1 else {})

    return loss_fn


def feature_metrics(pred, target):
    return {
        "mse": float(F.mse_loss(pred.float(), target.float()).detach().cpu()),
        "cos": float(F.cosine_similarity(pred.float().flatten(1), target.float().flatten(1), dim=1).mean().detach().cpu()),
    }
