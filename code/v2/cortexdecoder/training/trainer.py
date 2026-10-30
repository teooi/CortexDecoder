import gc
import json
import math
import os
import random
from collections import defaultdict

import pandas as pd
import torch
from tqdm.auto import tqdm

from ..core.common import AMP_DTYPE, DEVICE, USE_AMP, noop, torch_load_cpu
from ..core.config import SPLIT_POLICY, VGG_IMAGE_SIZE
from .losses import build_clip_loss, feature_metrics, sample_layer_indices, select_target_tokens, vgg_contrastive_loss
from ..models.bridge import build_bridge


def head_paths(out_dir, head):
    return {
        "best": f"{out_dir}/{head}_best.pt",
        "latest": f"{out_dir}/{head}_latest.pt",
        "history": f"{out_dir}/train_{head}_history.json",
        "history_csv": f"{out_dir}/train_{head}_history.csv",
    }


def architecture_name(cfg, head):
    return f"domain_specialized_{head}_single_head_from_scratch_{cfg.subject_tag}_baseline_direct_flatmap"


def assert_checkpoint_compatible(ckpt, path, cfg, head):
    if not isinstance(ckpt, dict):
        raise ValueError(f"refusing to load legacy bare checkpoint without split metadata: {path}")
    if ckpt.get("split_policy") != SPLIT_POLICY:
        raise ValueError(f"refusing to load checkpoint with split_policy={ckpt.get('split_policy')!r}; expected {SPLIT_POLICY!r}: {path}")
    if float(ckpt.get("val_frac", cfg.val_frac)) != float(cfg.val_frac):
        raise ValueError(f"refusing to load checkpoint with val_frac={ckpt.get('val_frac')!r}; expected {cfg.val_frac!r}: {path}")
    if "direct_flatmap" not in str(ckpt.get("architecture", "")):
        raise ValueError(f"refusing to load checkpoint with architecture={ckpt.get('architecture')!r}: {path}")
    if ckpt.get("domain") != head:
        raise ValueError(f"checkpoint domain={ckpt.get('domain')!r} does not match head={head!r}: {path}")
    # v1 checkpoints predate these keys; they were all trained with the defaults.
    defaults = {"n_layers": 4, "n_heads": 8, "clip_loss": "mse", "clip_temperature": 0.07}
    keys = ["n_layers", "n_heads"] + (["clip_loss", "clip_temperature"] if head == "clip" else [])
    for key in keys:
        if ckpt.get(key, defaults[key]) != getattr(cfg, key):
            raise ValueError(f"checkpoint {key}={ckpt.get(key, defaults[key])!r} does not match config {getattr(cfg, key)!r}: {path}")


def load_checkpoint_into(model, path, cfg, head):
    ckpt = torch_load_cpu(path, map_location=DEVICE)
    assert_checkpoint_compatible(ckpt, path, cfg, head)
    model.load_state_dict(ckpt["model"])
    return ckpt


def checkpoint_payload(cfg, data, head, model, opt, scheduler, epoch, best_val):
    return {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "domain": head,
        "source_checkpoint": None,
        "initialization": "scratch",
        "architecture": architecture_name(cfg, head),
        "subjects": [cfg.subject],
        "num_flatmap_patches": data.num_patches,
        "architecture_mode": "baseline",
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "clip_loss": cfg.clip_loss,
        "clip_temperature": cfg.clip_temperature,
        "train_epochs": int(cfg.epochs),
        "train_lr": cfg.lr,
        "split_policy": SPLIT_POLICY,
        "val_frac": cfg.val_frac,
        "final_heldout_image_ids_count": len(data.final_heldout_ids),
        "val_heldout_image_ids_by_subject_count": {cfg.subject: len(data.val_heldout_ids)},
        "val_heldout_image_ids_all_count": len(data.val_heldout_ids),
    }


def set_warmup_lr(cfg, opt, epoch):
    if cfg.warmup_epochs <= 0 or epoch > cfg.warmup_epochs:
        return
    for group in opt.param_groups:
        group["lr"] = cfg.lr * max(epoch, 1) / cfg.warmup_epochs


class HeadTrainer:
    """Trains one bridge head (VGG or CLIP) from scratch, resuming from `<head>_latest.pt` when present."""

    def __init__(self, cfg, data, head, out_dir, stimuli=None, tokenizer=None):
        if head == "vgg" and (stimuli is None or tokenizer is None):
            raise ValueError("VGG training needs stimuli and a VGG tokenizer")
        self.cfg, self.data, self.head, self.out_dir = cfg, data, head, out_dir
        self.stimuli, self.tokenizer = stimuli, tokenizer
        self.paths = head_paths(out_dir, head)
        self.model = build_bridge(cfg, data, head, DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=cfg.plateau_factor, patience=cfg.plateau_patience,
        )
        print(f"{head} single-head training params: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
        if head == "clip":
            self.clip_loss_fn = build_clip_loss(cfg)

    def batch_forward(self, brain_batch, id_batch, train):
        """Returns (pred, target, loss, loss_parts)."""
        if self.head == "vgg":
            images = self.stimuli.for_ids(id_batch, size=VGG_IMAGE_SIZE).to(DEVICE, non_blocking=True)
            with torch.no_grad():
                target_layers = self.tokenizer(images)
            query_ids, splits = sample_layer_indices(DEVICE, train=train)
            target = select_target_tokens(target_layers, query_ids, splits)
            with torch.set_grad_enabled(train), torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                pred = self.model(brain_batch, query_indices=query_ids)
                loss, parts = vgg_contrastive_loss(pred, target, splits, self.cfg.temperature)
            return pred, target, loss, parts
        target = self.data.clip_targets_for_ids(id_batch).to(DEVICE, non_blocking=True).float()
        with torch.set_grad_enabled(train), torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
            pred = self.model(brain_batch)
            loss, parts = self.clip_loss_fn(pred, target)
        return pred, target, loss, parts

    def run_epoch(self, epoch, train=True):
        cfg = self.cfg
        self.model.train(train)
        split = "train" if train else "val"
        order = list(self.data.sessions)
        if train:
            random.shuffle(order)
            set_warmup_lr(cfg, self.opt, epoch)
        totals = defaultdict(float)
        n_batches = 0
        progress = tqdm(
            total=self.data.expected_batches(split, cfg.batch_size),
            desc=f"[train {self.head}] epoch {epoch}/{cfg.epochs} ({split})",
            leave=False, unit="batch", mininterval=10.0,
        )
        try:
            for sess in order:
                loaded = self.data.load_session(sess, split=split)
                if loaded is None:
                    continue
                brain_tokens, img_ids = loaded
                perm = torch.randperm(brain_tokens.shape[0]) if train else torch.arange(brain_tokens.shape[0])
                for b0 in range(0, len(perm), cfg.batch_size):
                    idx = perm[b0:b0 + cfg.batch_size]
                    brain_batch = brain_tokens[idx].to(DEVICE, non_blocking=True).float()
                    pred, target, loss, parts = self.batch_forward(brain_batch, img_ids[idx], train)
                    if train:
                        self.opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
                        self.opt.step()
                    metrics = feature_metrics(pred, target)
                    totals["loss"] += float(loss.detach().cpu())
                    totals["mse"] += metrics["mse"]
                    totals["cos"] += metrics["cos"]
                    totals["lr"] += float(self.opt.param_groups[0]["lr"])
                    for key, value in parts.items():
                        totals[key] += float(value.detach().cpu())
                    n_batches += 1
                    progress.update(1)
                    progress.set_postfix(sess=int(sess), loss=f"{float(loss.detach().cpu()):.4f}")
                if not cfg.cache_sessions_in_ram:
                    del brain_tokens, img_ids
                    gc.collect()
        finally:
            progress.close()
        if n_batches == 0:
            raise RuntimeError(f"no valid batches for {self.head} epoch; bad sessions={sorted(self.data.bad_sessions)}")
        return {k: v / n_batches for k, v in totals.items()}

    def audit_vgg_batch(self):
        """Gradient sanity check on one training batch before VGG training starts."""
        for sess in self.data.sessions:
            loaded = self.data.load_session(sess, split="train")
            if loaded is not None and loaded[0].shape[0] >= 2:
                break
        else:
            raise RuntimeError("no usable debug batch for split='train'")
        n = min(self.cfg.batch_size, 64, loaded[0].shape[0])
        brain_batch = loaded[0][:n].to(DEVICE, non_blocking=True).float()
        id_batch = loaded[1][:n]
        self.opt.zero_grad(set_to_none=True)
        pred, target, loss, _ = self.batch_forward(brain_batch, id_batch, train=True)
        loss.backward()
        grads = [p.grad.detach().float().square().sum() for p in self.model.parameters() if p.grad is not None]
        grad_norm = math.sqrt(float(sum(grads).cpu())) if grads else 0.0
        self.opt.zero_grad(set_to_none=True)
        metrics = feature_metrics(pred.detach(), target.detach())
        print(
            f"VGG batch audit sess={sess} batch={n} unique_ids={len(set(id_batch.tolist()))} "
            f"loss={float(loss.detach().cpu()):.4f} chance_log_batch={math.log(max(n, 2)):.4f} cos={metrics['cos']:.4f} "
            f"pred_std={float(pred.detach().float().std().cpu()):.4f} target_std={float(target.detach().float().std().cpu()):.4f} "
            f"grad_norm={grad_norm:.4e} grad_tensors={len(grads)}"
        )

    def _resume(self):
        history, best_val, start_epoch = [], float("inf"), 1
        if os.path.exists(self.paths["history"]):
            with open(self.paths["history"]) as f:
                history = json.load(f)
            if history:
                best_val = min(h["val_loss"] for h in history)
                start_epoch = max(h["epoch"] for h in history) + 1
        if os.path.exists(self.paths["latest"]):
            ckpt = load_checkpoint_into(self.model, self.paths["latest"], self.cfg, self.head)
            self.opt.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = max(start_epoch, int(ckpt["epoch"]) + 1)
            best_val = min(best_val, float(ckpt["best_val"]))
            print(f"resuming {self.head} training from {self.paths['latest']}; next epoch={start_epoch}")
        else:
            print(f"starting {self.head} training from fresh random initialization")
        return history, best_val, start_epoch

    def _save(self, epoch, best_val, history, is_best):
        payload = checkpoint_payload(self.cfg, self.data, self.head, self.model, self.opt, self.scheduler, epoch, best_val)
        torch.save(payload, self.paths["latest"])
        if is_best:
            torch.save(payload, self.paths["best"])
        with open(self.paths["history"], "w") as f:
            json.dump(history, f, indent=2)
        if history:
            pd.DataFrame(history).to_csv(self.paths["history_csv"], index=False)

    def fit(self, on_epoch_end=noop):
        cfg = self.cfg
        history, best_val, start_epoch = self._resume()
        if self.head == "vgg" and cfg.run_vgg_batch_audit:
            self.audit_vgg_batch()
        if start_epoch > cfg.epochs:
            print(f"{self.head} training already complete for epochs={cfg.epochs}; loaded {len(history)} rows")
            return history
        for epoch in range(start_epoch, cfg.epochs + 1):
            train_scores = self.run_epoch(epoch, train=True)
            val_scores = self.run_epoch(epoch, train=False)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_scores.items()}, **{f"val_{k}": v for k, v in val_scores.items()}}
            history.append(row)
            improved = row["val_loss"] < best_val
            if improved:
                best_val = row["val_loss"]
            if epoch > cfg.warmup_epochs:
                self.scheduler.step(row["val_loss"])
            self._save(epoch, best_val, history, improved)
            on_epoch_end()
            print(
                f"{self.head} epoch {epoch:02d} lr={self.opt.param_groups[0]['lr']:.2e} "
                f"train_loss={row['train_loss']:.4f} train_cos={row['train_cos']:.4f} "
                f"val_loss={row['val_loss']:.4f} val_cos={row['val_cos']:.4f} best_val={best_val:.4f}"
            )
        return history
