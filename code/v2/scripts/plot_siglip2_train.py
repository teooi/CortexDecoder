"""SigLIP2-conditioned SDXL unCLIP fine-tune: diffusion loss vs. training updates, same paper style as plot_loss_sweep.py.

    .venv/bin/python code/v2/scripts/plot_siglip2_train.py <train_siglip2_unclip.log> <train_siglip2_unclip_history.csv> [out_stem]

Train loss is the mean of every per-update loss in each 1,000-update window (parsed from the log); the history CSV's
train_loss is a single batch and too noisy to plot. Validation loss comes from the history CSV.
"""
import csv
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from plot_loss_sweep import save, style_axes  # shared rcParams (9 pt Times, TrueType fonts), axes style, saving

WINDOW = 1000
UPDATE = re.compile(r"UPDATE ([\d,]+)/[\d,]+ \| loss=([0-9.eE+-]+)")


def train_curve(log_path):
    sums, counts = defaultdict(float), defaultdict(int)
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = UPDATE.search(line)
            if m:
                step = int(m.group(1).replace(",", ""))
                bucket = (step - 1) // WINDOW + 1
                sums[bucket] += float(m.group(2))
                counts[bucket] += 1
    buckets = sorted(b for b in sums if counts[b] >= WINDOW // 2)  # skip a trailing partial window
    return np.array(buckets) * WINDOW, np.array([sums[b] / counts[b] for b in buckets])


def val_curve(history_path):
    with open(history_path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("val_loss")]
    return np.array([int(float(r["step"])) for r in rows]), np.array([float(r["val_loss"]) for r in rows])


def main(log_path, history_path, out_stem):
    t_steps, t_loss = train_curve(log_path)
    v_steps, v_loss = val_curve(history_path)

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    style_axes(ax)
    ax.plot(t_steps / 1000, t_loss, color="#7FB0DC", linewidth=0.9, label="Train (1k-update mean)", zorder=2)
    ax.plot(v_steps / 1000, v_loss, color="#2F6FA8", linewidth=1.1, marker="o", markersize=2.6,
            markevery=max(len(v_steps) // 16, 1), markerfacecolor="#AED2EF", markeredgecolor="#2F6FA8",
            markeredgewidth=0.6, label="Validation", zorder=3)
    best = int(np.argmin(v_loss))
    ax.plot(v_steps[best] / 1000, v_loss[best], marker="*", markersize=6.5, color="#2F6FA8",
            markeredgecolor="black", markeredgewidth=0.4, linestyle="none", zorder=4)

    ax.set_xlabel("Training updates (k)")
    ax.set_ylabel("Diffusion loss")
    ax.set_xlim(0, max(t_steps.max(), v_steps.max()) / 1000 * 1.02)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=2,
               frameon=False, handlelength=2.2, columnspacing=1.0, handletextpad=0.5)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.86)
    save(fig, out_stem)
    print(f"train windows={len(t_steps)} last={t_steps[-1]} | val points={len(v_steps)} "
          f"best val={v_loss[best]:.5f} @ {v_steps[best]} | last val={v_loss[-1]:.5f} @ {v_steps[-1]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "papers/AAAI_workshop_paper/Figures/siglip2_unclip_loss")
