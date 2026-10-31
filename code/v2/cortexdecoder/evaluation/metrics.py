import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim_fn
from tqdm.auto import tqdm

from ..core.common import DEVICE, torch_load_cpu
from ..data.nsd import final_eval_indices, rows_for_ids
from ..core.figures import plt, save_figure

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]


# --- predicted vs actual CLIP embeddings (semantic branch, before diffusion) ---

def clip_embedding_eval(cfg, data, prediction, out_dir):
    ids = [int(i) for i in prediction["shared_img_ids"]]
    pred_all = prediction["pred_clip_tokens"].float()
    true_all = data.clip_targets_for_ids(ids).float()
    idx = final_eval_indices(cfg, min(len(ids), pred_all.shape[0], true_all.shape[0]))
    pred, true = pred_all[idx].contiguous(), true_all[idx].contiguous()

    image_cos = F.cosine_similarity(pred.flatten(1), true.flatten(1), dim=1).numpy()
    token_cos = F.cosine_similarity(pred, true, dim=-1).mean(dim=1).numpy()
    image_mse = ((pred - true) ** 2).flatten(1).mean(dim=1).numpy()
    # Cheap proxies for reconstruction CLIP/Inception: can the predicted tokens tell images apart?
    sim = _pearson_matrix(pred.flatten(1), true.flatten(1))
    retr_img, retr_brain = retrieval_score(pred.flatten(1), true.flatten(1), seed=cfg.seed, n_samples=min(300, len(idx)))
    row = {
        "subject": cfg.subject,
        "N": len(idx),
        "token_two_way": float((sim < sim.diag()[None, :]).float().sum(0).mean() / (len(idx) - 1)),
        "token_retrieval_img": retr_img,
        "token_retrieval_brain": retr_brain,
        "clip_mse": float(np.mean(image_mse)),
        "clip_cos_flat_mean": float(np.mean(image_cos)),
        "clip_cos_flat_median": float(np.median(image_cos)),
        "clip_cos_token_mean": float(np.mean(token_cos)),
        "clip_cos_token_median": float(np.median(token_cos)),
        "split_policy": "shared1000_image_id_holdout",
        "target_source": "cached_actual_clip_embeddings",
    }
    pd.DataFrame({
        "subject": cfg.subject,
        "img_id": [ids[i] for i in idx],
        "clip_mse": image_mse,
        "clip_cos_flat": image_cos,
        "clip_cos_token_mean": token_cos,
    }).to_csv(f"{out_dir}/{cfg.subject}_shared1000_clip_embedding_eval.csv", index=False)
    summary = pd.DataFrame([row])
    summary.to_csv(f"{out_dir}/shared1000_clip_embedding_eval_summary.csv", index=False)
    print(summary.to_string(index=False))

    hist_dir = f"{out_dir}/clip_embedding_eval_hists"
    os.makedirs(hist_dir, exist_ok=True)
    fig = plt.figure(figsize=(7, 4))
    plt.hist(image_cos, bins=40)
    plt.title(f"{cfg.subject} predicted vs actual CLIP cosine")
    plt.xlabel("cosine similarity")
    plt.ylabel("images")
    plt.tight_layout()
    save_figure(fig, f"{hist_dir}/{cfg.subject}_clip_cos_flat_hist.png")
    return row


def _pearson_matrix(a, b):
    """[i, j] = Pearson r between row i of `a` (preds) and row j of `b` (targets)."""
    a = F.normalize(a.float() - a.float().mean(1, keepdim=True), dim=1)
    b = F.normalize(b.float() - b.float().mean(1, keepdim=True), dim=1)
    return b @ a.T  # rows: targets, cols: preds (same layout as two_way_score)


# --- final reconstruction metrics (MindEye2 table) ---------------------------

def norm_batch(x, size, mean, std):
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def two_way_score(recon_feats, true_feats):
    r = np.corrcoef(true_feats.numpy(), recon_feats.numpy())[:len(true_feats), len(true_feats):]
    congruent = np.diag(r)
    return float(np.mean(np.sum(r < congruent, axis=0)) / (len(true_feats) - 1))


def correlation_distance(recon_feats, true_feats):
    a, b = recon_feats.numpy(), true_feats.numpy()
    return float(np.mean([1.0 - np.corrcoef(a[i], b[i])[0, 1] for i in range(len(a))]))


def retrieval_score(pred_feats, true_feats, seed, n_loops=30, n_samples=300):
    pred = F.normalize(pred_feats.float(), dim=-1)
    true = F.normalize(true_feats.float(), dim=-1)
    rng = np.random.RandomState(seed)
    fwd, bwd = [], []
    for _ in range(n_loops):
        idx = torch.as_tensor(rng.choice(len(pred), size=min(n_samples, len(pred)), replace=False), dtype=torch.long)
        labels = torch.arange(len(idx))
        fwd.append((pred[idx] @ true[idx].T).argmax(1).eq(labels).float().mean().item())
        bwd.append((true[idx] @ pred[idx].T).argmax(1).eq(labels).float().mean().item())
    return float(np.mean(fwd)), float(np.mean(bwd))


def pixel_metrics(recons, images):
    recons = F.interpolate(recons, size=(425, 425), mode="bilinear", align_corners=False)
    images = F.interpolate(images, size=(425, 425), mode="bilinear", align_corners=False)
    pixcorr = np.mean([np.corrcoef(a.flatten(), b.flatten())[0, 1] for a, b in zip(recons.numpy(), images.numpy())])
    rec_gray = rgb2gray(recons.permute(0, 2, 3, 1).numpy())
    img_gray = rgb2gray(images.permute(0, 2, 3, 1).numpy())
    ssim = np.mean([
        ssim_fn(r, i, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        for r, i in zip(rec_gray, img_gray)
    ])
    return float(pixcorr), float(ssim)


class FeatureMetrics:
    """Frozen AlexNet / InceptionV3 / EfficientNet-B1 / CLIP ViT-L-14 / SwAV feature extractors."""

    def __init__(self, cfg):
        import open_clip
        from torchvision.models import (
            AlexNet_Weights, EfficientNet_B1_Weights, Inception_V3_Weights, alexnet, efficientnet_b1, inception_v3,
        )
        from torchvision.models.feature_extraction import create_feature_extractor

        def frozen(m):
            return m.to(DEVICE).eval().requires_grad_(False)

        self.batch_size = cfg.eval_feature_batch_size
        self.alex = frozen(create_feature_extractor(
            alexnet(weights=AlexNet_Weights.IMAGENET1K_V1), return_nodes={"features.4": "a2", "features.11": "a5"},
        ))
        self.inception = frozen(create_feature_extractor(inception_v3(weights=Inception_V3_Weights.DEFAULT), return_nodes={"avgpool": "pool"}))
        self.effnet = frozen(create_feature_extractor(efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT), return_nodes={"avgpool": "pool"}))
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=DEVICE)
        self.clip = frozen(clip_model)
        self.swav = None
        if cfg.run_swav_eval:
            self.swav = frozen(create_feature_extractor(torch.hub.load("facebookresearch/swav:main", "resnet50"), return_nodes={"avgpool": "pool"}))

    @torch.no_grad()
    def _extract(self, images, fn):
        outs = []
        for start in tqdm(range(0, len(images), self.batch_size), desc="feature batches", leave=False):
            outs.append(fn(images[start:start + self.batch_size].to(DEVICE)).float().flatten(1).cpu())
        return torch.cat(outs, dim=0)

    def all(self, x):
        feats = {
            "alex2": self._extract(x, lambda b: self.alex(norm_batch(b, 256, IMAGENET_MEAN, IMAGENET_STD))["a2"]),
            "alex5": self._extract(x, lambda b: self.alex(norm_batch(b, 256, IMAGENET_MEAN, IMAGENET_STD))["a5"]),
            "inception": self._extract(x, lambda b: self.inception(norm_batch(b, 342, IMAGENET_MEAN, IMAGENET_STD))["pool"]),
            "clip": self._extract(x, lambda b: self.clip.encode_image(norm_batch(b, 224, CLIP_MEAN, CLIP_STD))),
            "eff": self._extract(x, lambda b: self.effnet(norm_batch(b, 255, IMAGENET_MEAN, IMAGENET_STD))["pool"]),
        }
        if self.swav is not None:
            feats["swav"] = self._extract(x, lambda b: self.swav(norm_batch(b, 224, IMAGENET_MEAN, IMAGENET_STD))["pool"])
        return feats


def save_image_exports(export_root, subject, img_ids, true_images, low_images, final_images):
    from torchvision.utils import save_image

    dirs = {
        "actual": f"{export_root}/{subject}/actual",
        "vgg_dip": f"{export_root}/{subject}/vgg_dip_preview",
        "final_diffusion": f"{export_root}/{subject}/final_diffusion",
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    for row, img_id in enumerate(img_ids):
        stem = f"{row:04d}_img{int(img_id):06d}"
        save_image(true_images[row].clamp(0, 1), f"{dirs['actual']}/{stem}.png")
        save_image(low_images[row].clamp(0, 1), f"{dirs['vgg_dip']}/{stem}.png")
        save_image(final_images[row].clamp(0, 1), f"{dirs['final_diffusion']}/{stem}.png")
    return f"{export_root}/{subject}"


def save_preview_grid(cfg, img_ids, true_images, low_images, final_images, indices, path):
    indices = [int(i) for i in indices]
    if not indices:
        return None
    low = F.interpolate(low_images[indices], size=final_images.shape[-1], mode="bicubic", align_corners=False)
    combined = torch.cat([true_images[indices], low, final_images[indices]], dim=3).clamp(0, 1)
    cols = min(cfg.preview_grid_cols, len(indices))
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 3.0 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for k, idx in enumerate(indices):
        ax = axes[k // cols, k % cols]
        ax.imshow(combined[k].permute(1, 2, 0))
        ax.set_title(f"{cfg.subject} id {img_ids[idx]}: true | low | final", fontsize=8)
    return save_figure(fig, path, bbox_inches="tight")


def evaluate_final(cfg, data, stimuli, diffusion_manifest, prediction, out_dir):
    subject, tag = cfg.subject, diffusion_manifest["final_recon_tag"]
    final_uint8 = torch_load_cpu(diffusion_manifest["final_recons_path"])
    with open(diffusion_manifest["final_ids_path"]) as f:
        img_ids = [int(i) for i in json.load(f)][:len(final_uint8)]
    if any(i not in data.final_heldout_ids for i in img_ids):
        raise ValueError(f"{subject} final eval IDs include non-heldout image IDs")

    true_images = stimuli.for_ids(img_ids, size=256).float()
    final_images = final_uint8.float().div(255.0)
    if final_images.shape[-1] != 256:
        final_images = F.interpolate(final_images, size=(256, 256), mode="bicubic", align_corners=False).clamp(0, 1)
    low_all = torch_load_cpu(diffusion_manifest["lowlevel_recons_path"]).float().clamp(0, 1)
    with open(diffusion_manifest["lowlevel_ids_path"]) as f:
        low_ids = [int(i) for i in json.load(f)]
    low_images = low_all[rows_for_ids(low_ids, img_ids, "low-level DIP images")]
    low_images = F.interpolate(low_images, size=(256, 256), mode="bicubic", align_corners=False).clamp(0, 1)

    export_root = f"{out_dir}/drive_image_exports_{tag}"
    print("saved per-image PNG exports:", save_image_exports(export_root, subject, img_ids, true_images, low_images, final_images))

    metrics = FeatureMetrics(cfg)
    true_feats = metrics.all(true_images)
    rows = []
    for method, recons in [("Low-Level", low_images), ("Final", final_images)]:
        recon_feats = metrics.all(recons)
        pixcorr, ssim = pixel_metrics(recons.cpu(), true_images.cpu())
        row = {
            "Subject": subject,
            "Method": method,
            "N": len(recons),
            "PixCorr": pixcorr,
            "SSIM": ssim,
            "AlexNet(2)": two_way_score(recon_feats["alex2"], true_feats["alex2"]),
            "AlexNet(5)": two_way_score(recon_feats["alex5"], true_feats["alex5"]),
            "InceptionV3": two_way_score(recon_feats["inception"], true_feats["inception"]),
            "CLIP": two_way_score(recon_feats["clip"], true_feats["clip"]),
            "EffNet-B": correlation_distance(recon_feats["eff"], true_feats["eff"]),
        }
        if "swav" in true_feats:
            row["SwAV"] = correlation_distance(recon_feats["swav"], true_feats["swav"])
        row["ImageRetrieval"], row["BrainRetrieval"] = retrieval_score(
            recon_feats["clip"], true_feats["clip"], seed=cfg.seed, n_samples=min(300, len(recons)),
        )
        rows.append(row)

    # Retrieval directly on predicted vs true OpenCLIP tokens (mean-pooled), independent of diffusion.
    pred_rows = rows_for_ids(prediction["shared_img_ids"], img_ids, "predicted CLIP tokens")
    pred_semantic = prediction["pred_clip_tokens"][pred_rows].float().mean(1)
    true_semantic = data.clip_targets_for_ids(img_ids).float().mean(1)
    rows[-1]["SemanticImageRetrieval"], rows[-1]["SemanticBrainRetrieval"] = retrieval_score(
        pred_semantic, true_semantic, seed=cfg.seed, n_samples=min(300, len(pred_semantic)),
    )

    df = pd.DataFrame(rows)
    df.set_index("Method").to_csv(f"{out_dir}/{subject}_mindeye2_table_metrics_{tag}.csv")
    df.to_csv(f"{out_dir}/mindeye2_table_metrics_summary_{tag}.csv", index=False)
    print(df.to_string(index=False))

    preview_dir = f"{out_dir}/brainit_collage_previews_{tag}"
    os.makedirs(preview_dir, exist_ok=True)
    n = len(final_images)
    rng = np.random.RandomState(cfg.seed)
    sample_idx = rng.choice(n, size=min(cfg.preview_grid_n, n), replace=False)
    save_preview_grid(cfg, img_ids, true_images, low_images, final_images, sample_idx,
                      f"{preview_dir}/{subject}_brainit_sample48_preview_{tag}.png")
    for part, start in enumerate(range(0, n, cfg.preview_grid_n), start=1):
        stop = min(start + cfg.preview_grid_n, n)
        save_preview_grid(cfg, img_ids, true_images, low_images, final_images, range(start, stop),
                          f"{preview_dir}/{subject}_brainit_collage_preview_{tag}_part{part:03d}_{start:04d}-{stop - 1:04d}.png")
    print("saved preview collages:", preview_dir)
    return df
