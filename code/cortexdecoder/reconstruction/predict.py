import gc
import json
import os

import torch
from tqdm.auto import tqdm

from ..core.common import AMP_DTYPE, DEVICE, USE_AMP, torch_load_cpu
from ..core.config import VGG_LAYER_TOKEN_COUNTS, VGG_TOTAL_TOKENS
from ..data.nsd import shared_ids_path
from ..models.bridge import build_bridge
from ..training.trainer import load_checkpoint_into


def pred_tokens_path(out_dir, subject, head):
    return f"{out_dir}/{subject}_shared1000_transfer_pred_{head}_tokens.pt"


@torch.no_grad()
def predict_all_vgg_tokens(model, brain_batch, query_chunk):
    brain_batch = brain_batch.to(DEVICE, non_blocking=True).float()
    outs = []
    for start in range(0, VGG_TOTAL_TOKENS, query_chunk):
        ids = torch.arange(start, min(start + query_chunk, VGG_TOTAL_TOKENS), device=DEVICE)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
            outs.append(model(brain_batch, query_indices=ids).float().cpu())
    return torch.cat(outs, dim=1)


@torch.no_grad()
def predict_clip_tokens(model, brain_batch):
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
        return model(brain_batch.to(DEVICE, non_blocking=True).float()).float().cpu()


def split_all_vgg_tokens(x):
    out, start = [], 0
    for n in VGG_LAYER_TOKEN_COUNTS:
        out.append(x[:, start:start + n])
        start += n
    return out


def predict_shared1000(cfg, data, head, ckpt_path, out_dir, shared_dir=None):
    """Predict `head` tokens for every shared-1000 image with the weights in `ckpt_path`. Cached in `out_dir`.

    The trial-averaged shared-1000 flatmaps are cached in `shared_dir` (default `out_dir`)."""
    shared_tokens, shared_ids = data.collect_shared1000(shared_dir or out_dir)
    print(f"{cfg.subject} shared flatmap source tokens:", tuple(shared_tokens.shape), "ids:", len(shared_ids))
    path = pred_tokens_path(out_dir, cfg.subject, head)
    if os.path.exists(path):
        print(f"{cfg.subject}: predicted {head} exists:", path)
        return path

    model = build_bridge(cfg, data, head, DEVICE)
    load_checkpoint_into(model, ckpt_path, cfg, head)
    model.eval()
    print("loaded prediction checkpoint:", ckpt_path)
    preds = []
    for start in tqdm(range(0, len(shared_tokens), cfg.predict_batch), desc=f"predict shared-1000 {head} tokens", mininterval=10.0):
        brain_batch = shared_tokens[start:start + cfg.predict_batch]
        if head == "vgg":
            preds.append(predict_all_vgg_tokens(model, brain_batch, cfg.predict_query_chunk))
        else:
            preds.append(predict_clip_tokens(model, brain_batch))
    pred = torch.cat(preds, dim=0).half().contiguous()
    torch.save(pred, path)
    print(f"{cfg.subject}: saved predicted {head} tokens:", tuple(pred.shape), path)
    del model, preds, pred, shared_tokens
    gc.collect()
    return path


def best_or_latest_checkpoint(out_dir, head):
    candidates = [f"{out_dir}/{head}_best.pt", f"{out_dir}/{head}_latest.pt"]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no checkpoint available for {head}; checked {candidates}")


def load_predictions(subject, shared_dir, head_dirs):
    """head_dirs: {head: dir holding that head's predicted tokens}."""
    with open(shared_ids_path(shared_dir, subject)) as f:
        ids = json.load(f)
    out = {"subject": subject, "shared_img_ids": ids}
    for head, d in head_dirs.items():
        out[f"pred_{head}_tokens"] = torch_load_cpu(pred_tokens_path(d, subject, head))
    return out


def write_manifest(path, manifest):
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
