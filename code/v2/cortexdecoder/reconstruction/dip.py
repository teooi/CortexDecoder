import gc
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim_fn
from tqdm.auto import tqdm

from ..core.common import AMP_DTYPE, DEVICE, noop, torch_load_cpu
from ..core.config import SHARED1000_EVAL_SIZE, VGG_IMAGE_SIZE
from ..data.nsd import final_eval_indices
from ..core.figures import plt, save_figure
from .predict import split_all_vgg_tokens


class GroupedConvBlock(nn.Module):
    def __init__(self, groups, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(groups * in_ch, groups * out_ch, 3, padding=1, groups=groups),
            nn.BatchNorm2d(groups * out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(groups * out_ch, groups * out_ch, 3, padding=1, groups=groups),
            nn.BatchNorm2d(groups * out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BatchedDIPUNet(nn.Module):
    """`groups` independent DIP U-Nets run as one grouped-conv network, one per image."""

    def __init__(self, groups, input_dim, width):
        super().__init__()
        self.groups = int(groups)
        self.down1 = GroupedConvBlock(groups, input_dim, width)
        self.down2 = GroupedConvBlock(groups, width, width)
        self.down3 = GroupedConvBlock(groups, width, width)
        self.mid = GroupedConvBlock(groups, width, width)
        self.up3 = GroupedConvBlock(groups, width * 2, width)
        self.up2 = GroupedConvBlock(groups, width * 2, width)
        self.up1 = GroupedConvBlock(groups, width * 2, width)
        self.out = nn.Conv2d(groups * width, groups * 3, 1, groups=groups)

    def _merge(self, x):
        b, c, h, w = x.shape
        if b != self.groups:
            raise ValueError(f"expected batch={self.groups}, got {b}")
        return x.reshape(1, b * c, h, w)

    def _split(self, x, channels):
        _, _, h, w = x.shape
        return x.reshape(self.groups, channels, h, w)

    def forward(self, z):
        z = self._merge(z)
        e1 = self.down1(z)
        e2 = self.down2(F.avg_pool2d(e1, 2))
        e3 = self.down3(F.avg_pool2d(e2, 2))
        x = self.mid(F.avg_pool2d(e3, 2))
        x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, e3], dim=1))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self._split(torch.sigmoid(self.out(x)), 3)


def reduce_repeated_token_dim(tokens, true_dim):
    if tokens.shape[-1] == true_dim:
        return tokens
    if tokens.shape[-1] % true_dim != 0:
        return tokens[..., :true_dim]
    return tokens.reshape(*tokens.shape[:-1], tokens.shape[-1] // true_dim, true_dim).mean(dim=-2)


def fold_overlapping_tokens(tokens, channels, output_hw, kernel_size, stride):
    cols = tokens.transpose(1, 2).contiguous()
    feat = F.fold(cols, output_size=output_hw, kernel_size=kernel_size, stride=stride)
    ones = torch.ones((1, channels * kernel_size * kernel_size, tokens.shape[1]), device=tokens.device, dtype=tokens.dtype)
    counts = F.fold(ones, output_size=output_hw, kernel_size=kernel_size, stride=stride).clamp_min(1e-6)
    return feat / counts


def tokens_to_brainit_vgg_features(tokens):
    """Invert BrainITVGGTokenizer: predicted [B, 7190, 512] tokens -> five VGG feature maps."""
    layers = split_all_vgg_tokens(tokens.float())
    feat1_tokens = reduce_repeated_token_dim(layers[0], 4 * 64)
    feat2_tokens = reduce_repeated_token_dim(layers[1], 4 * 128)
    feat3_tokens = reduce_repeated_token_dim(layers[2], 256)
    feat4_tokens = reduce_repeated_token_dim(layers[3], 512)
    feat5_tokens = reduce_repeated_token_dim(layers[4], 512)
    b = tokens.shape[0]
    return [
        fold_overlapping_tokens(feat1_tokens, channels=64, output_hw=(112, 112), kernel_size=2, stride=2),
        fold_overlapping_tokens(feat2_tokens, channels=128, output_hw=(56, 56), kernel_size=2, stride=1),
        feat3_tokens.transpose(1, 2).reshape(b, 256, 28, 28).contiguous(),
        feat4_tokens.transpose(1, 2).reshape(b, 512, 14, 14).contiguous(),
        feat5_tokens.transpose(1, 2).reshape(b, 512, 7, 7).contiguous(),
    ]


def dip_loss(cfg, tokenizer, pred_img, target_features):
    loss = pred_img.new_tensor(0.0)
    for weight, pred, target in zip(cfg.dip_layer_weights, tokenizer.extract_features(pred_img), target_features):
        loss = loss + weight * F.mse_loss(pred.float(), target.float())
    return loss / sum(cfg.dip_layer_weights)


def invert_vgg_features_with_dip(cfg, tokenizer, target_features, generator, desc="DIP"):
    batch_size = int(target_features[0].shape[0])
    dip = BatchedDIPUNet(batch_size, cfg.dip_input_dim, cfg.dip_internal_dim).to(DEVICE).train()
    z = torch.randn(
        batch_size, cfg.dip_input_dim, cfg.dip_output_size, cfg.dip_output_size,
        device=DEVICE, generator=generator,
    ) * cfg.dip_initial_noise
    opt = torch.optim.Adam(dip.parameters(), lr=cfg.dip_lr)
    ema = None
    target_features = [t.to(DEVICE).float().detach() for t in target_features]

    pbar = tqdm(range(cfg.dip_steps), desc=desc, leave=False, mininterval=10.0)
    for step in pbar:
        z_noise = torch.randn(z.shape, device=DEVICE, dtype=z.dtype, generator=generator)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
            img = dip(z + z_noise * cfg.dip_reg_noise)
            loss = dip_loss(cfg, tokenizer, img, target_features)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            current = dip(z).detach()
            ema = current if ema is None else cfg.dip_ema * ema + (1.0 - cfg.dip_ema) * current
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

    lowlevel_112 = ema.clamp(0, 1)
    lowlevel_256 = F.interpolate(
        lowlevel_112, size=(cfg.dip_upsampled_size, cfg.dip_upsampled_size), mode="bicubic", align_corners=False,
    ).clamp(0, 1)
    return lowlevel_112.cpu(), lowlevel_256.cpu()


def _subset_tag(cfg):
    """Per-image shards are shared by every subset size; aggregates/evals are named by size so they never clobber."""
    tag = cfg.dip_output_tag
    return tag if cfg.final_eval_images == SHARED1000_EVAL_SIZE else f"{tag}_n{cfg.final_eval_images}"


def dip_paths(cfg, out_dir):
    tag, subject = _subset_tag(cfg), cfg.subject
    recon_dir = f"{out_dir}/{subject}_shared1000_dip_recon_shards_{cfg.dip_output_tag}"
    os.makedirs(recon_dir, exist_ok=True)
    return {
        "recon_dir": recon_dir,
        "recons_path": f"{out_dir}/{subject}_shared1000_dip_recons_{tag}.pt",
        "recons112_path": f"{out_dir}/{subject}_shared1000_dip_recons_112_{tag}.pt",
        "img_ids_path": f"{out_dir}/{subject}_shared1000_dip_img_ids_{tag}.json",
        "source_indices_path": f"{out_dir}/{subject}_shared1000_dip_source_indices_{tag}.json",
    }


def _shard_path(paths, i):
    return f"{paths['recon_dir']}/recon_{i:04d}.pt"


def run_dip_inversion(cfg, tokenizer, stimuli, prediction, out_dir, on_batch_end=noop):
    """Invert predicted (or ground-truth) VGG features to low-level images. Resumable via per-image shards."""
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(cfg.seed)
    shared_img_ids = prediction["shared_img_ids"]
    pred_vgg_tokens = prediction["pred_vgg_tokens"]
    paths = dip_paths(cfg, out_dir)
    print("DIP subject:", cfg.subject, "predicted VGG:", tuple(pred_vgg_tokens.shape), "ids:", len(shared_img_ids))

    n_available = min(SHARED1000_EVAL_SIZE, len(shared_img_ids), pred_vgg_tokens.shape[0])
    recon_indices = final_eval_indices(cfg, n_available)
    pending = [i for i in recon_indices if cfg.rerun_dip or not os.path.exists(_shard_path(paths, i))]
    batch_size = min(cfg.dip_batch_size, max(len(pending), 1))
    print(
        f"DIP images to invert: {len(pending)} pending / {len(recon_indices)} requested / {n_available} available, "
        f"batch_size={batch_size}, rerun={cfg.rerun_dip}, target={cfg.dip_target_source}"
    )

    def target_features_for(batch_indices):
        if cfg.dip_use_ground_truth_vgg:
            images = stimuli.for_ids([int(shared_img_ids[i]) for i in batch_indices], size=VGG_IMAGE_SIZE).to(DEVICE)
            with torch.no_grad():
                return [feat.detach().cpu() for feat in tokenizer.extract_features(images)]
        return tokens_to_brainit_vgg_features(pred_vgg_tokens[batch_indices])

    start = 0
    pbar = tqdm(total=len(pending), desc="DIP invert images")
    while start < len(pending):
        batch_indices = pending[start:start + batch_size]
        try:
            recon112, recon256 = invert_vgg_features_with_dip(
                cfg, tokenizer, target_features_for(batch_indices), generator,
                desc=f"DIP {cfg.dip_target_source} batch {start // batch_size + 1}",
            )
            for j, i in enumerate(batch_indices):
                torch.save({
                    "img_id": int(shared_img_ids[i]),
                    "source_index": int(i),
                    "recon112": recon112[j].half(),
                    "recon": recon256[j].half(),
                    "target_source": cfg.dip_target_source,
                }, _shard_path(paths, i))
            start += len(batch_indices)
            pbar.update(len(batch_indices))
            del recon112, recon256
            on_batch_end()
        except torch.cuda.OutOfMemoryError:
            if batch_size <= cfg.dip_min_batch_size:
                raise
            batch_size = max(cfg.dip_min_batch_size, batch_size // 2)
            print(f"CUDA OOM; retrying with DIP batch_size={batch_size}")
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    pbar.close()

    items = [torch_load_cpu(_shard_path(paths, i)) for i in recon_indices]
    all_recons112 = torch.stack([it["recon112"].float() for it in items]).clamp(0, 1)
    all_recons = torch.stack([it["recon"].float() for it in items]).clamp(0, 1)
    torch.save(all_recons112, paths["recons112_path"])
    torch.save(all_recons, paths["recons_path"])
    with open(paths["img_ids_path"], "w") as f:
        json.dump([int(it["img_id"]) for it in items], f)
    with open(paths["source_indices_path"], "w") as f:
        json.dump([int(it.get("source_index", i)) for it, i in zip(items, recon_indices)], f)
    print("saved DIP recons 112:", tuple(all_recons112.shape), paths["recons112_path"])
    print("saved DIP recons 256:", tuple(all_recons.shape), paths["recons_path"])
    manifest = {cfg.subject: {"subject": cfg.subject, "n_requested": len(recon_indices), "n_available": n_available, **paths}}
    with open(f"{out_dir}/dip_prediction_manifest_{_subset_tag(cfg)}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest[cfg.subject]


def eval_dip(cfg, stimuli, paths, out_dir):
    """PixCorr/SSIM of the DIP reconstructions plus an actual|DIP preview grid."""
    tag, subject = _subset_tag(cfg), cfg.subject
    all_recons = torch_load_cpu(paths["recons_path"]).float().clamp(0, 1)
    with open(paths["img_ids_path"]) as f:
        ids_done = json.load(f)[:len(all_recons)]
    true_images = stimuli.for_ids(ids_done, size=cfg.dip_upsampled_size).float()

    pixcorr_scores, ssim_scores = [], []
    for rec, true in zip(all_recons, true_images):
        rec_np = rec.permute(1, 2, 0).numpy()
        true_np = true.permute(1, 2, 0).numpy()
        pixcorr_scores.append(np.corrcoef(rec_np.reshape(-1), true_np.reshape(-1))[0, 1])
        ssim_scores.append(ssim_fn(rgb2gray(true_np), rgb2gray(rec_np), data_range=1.0))
    row = {
        "subject": subject,
        "N": len(all_recons),
        "Pixel_MSE": F.mse_loss(all_recons, true_images).item(),
        "PixCorr": float(np.nanmean(pixcorr_scores)),
        "SSIM": float(np.nanmean(ssim_scores)),
        "target_source": cfg.dip_target_source,
    }
    df = pd.DataFrame([row])
    df.to_csv(f"{out_dir}/{subject}_shared1000_dip_eval_{tag}.csv", index=False)
    df.to_csv(f"{out_dir}/final_dip_eval_summary_{tag}.csv", index=False)
    print(df.to_string(index=False))

    n_show = min(cfg.preview_grid_n, len(all_recons))
    cols = min(cfg.preview_grid_cols, n_show)
    rows = int(np.ceil(n_show / cols))
    combined = torch.cat([true_images[:n_show], all_recons[:n_show]], dim=3).clamp(0, 1)
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.0 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for k in range(n_show):
        ax = axes[k // cols, k % cols]
        ax.imshow(combined[k].permute(1, 2, 0))
        ax.set_title(f"{subject} shared id {ids_done[k]}\nactual | VGG-DIP", fontsize=8)
    preview_path = f"{out_dir}/{subject}_shared1000_dip_preview_grid_{tag}.png"
    save_figure(fig, preview_path, bbox_inches="tight")
    print("saved:", preview_path)
    return row
