"""Semantic-branch loss sweep figure, styled for the AAAI paper (single column, 9 pt Times, no Type-3 fonts).

    .venv/bin/python code/v2/scripts/plot_loss_sweep.py <dir> [out_stem]

<dir> holds each run's train_clip_history.csv as hist_<run>.csv and, when available, its
shared1000_clip_embedding_eval_summary.csv as eval_<run>.csv.
Top: validation token cosine per epoch. Bottom: test-set identification of the predicted tokens.
Writes <out_stem>.pdf (vector) and <out_stem>.png (600 dpi). --curves-only drops the bottom panel.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Colours from the paper's diagrams: semantic-feature blue, brain-token green, query-token orange.
# Lines/edges use a deeper shade, fills use the diagram colour; lambda=0.3 is dashed / hatched.
RUNS = [
    # (file suffix, label, line colour, face colour, linestyle, marker, hatch)
    ("_mse", "MSE", "#2F6FA8", "#AED2EF", "-", "o", ""),
    ("_loss-mse_0.1_infonce", r"MSE + 0.1$\,$InfoNCE", "#3F8E47", "#7DC17D", "-", "s", ""),
    ("_loss-mse_0.3_infonce", r"MSE + 0.3$\,$InfoNCE", "#3F8E47", "#7DC17D", "--", "s", "////"),
    ("_loss-mse_0.1_softclip_T-0.01", r"MSE + 0.1$\,$SoftCLIP", "#C9741F", "#F9C784", "-", "^", ""),
    ("_loss-mse_0.3_softclip_T-0.01", r"MSE + 0.3$\,$SoftCLIP", "#C9741F", "#F9C784", "--", "^", "////"),
]
METRICS = [("token_two_way", "Two-way"), ("token_retrieval_img", "Image retr."), ("token_retrieval_brain", "Brain retr.")]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",  # Times-compatible math
    "font.size": 9,  # AAAI: text inside figures must be at least 9 pt
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.6,
    "hatch.linewidth": 0.5,
    "pdf.fonttype": 42,  # embed TrueType, never Type 3
    "ps.fonttype": 42,
})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def style_axes(ax, minor_x=True):
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.minorticks_on()
    if not minor_x:
        ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", direction="out", width=0.6)
    ax.tick_params(which="major", length=3)
    ax.tick_params(which="minor", length=1.5)


def main(src, out_stem, curves_only=False):
    src = Path(src)
    if curves_only:
        fig, ax_curve = plt.subplots(figsize=(3.3, 2.6))
        ax_bar = None
    else:
        fig, (ax_curve, ax_bar) = plt.subplots(2, 1, figsize=(3.3, 4.7), gridspec_kw={"height_ratios": [1.15, 1.0]})

    # --- top: validation cosine per epoch ---
    style_axes(ax_curve)
    for suffix, label, line, face, style, marker, _ in RUNS:
        path = src / f"hist{suffix}.csv"
        if not path.exists():
            continue
        rows = read_csv(path)
        epochs = [int(float(r["epoch"])) for r in rows]
        val_cos = [float(r["val_cos"]) for r in rows]
        ax_curve.plot(epochs, val_cos, label=label, color=line, linestyle=style, linewidth=1.1,
                      marker=marker, markersize=3.2, markevery=2,
                      markerfacecolor=face, markeredgecolor=line, markeredgewidth=0.6, zorder=3)
        best = int(np.argmax(val_cos))
        ax_curve.plot(epochs[best], val_cos[best], marker="*", markersize=6.5, color=line,
                      markeredgecolor="black", markeredgewidth=0.4, linestyle="none", zorder=4)
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Val. token cosine")
    ax_curve.set_xlim(0.5, 25.5)

    if curves_only:
        fig.legend(*ax_curve.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.54, 1.0),
                   ncol=2, frameon=False, handlelength=2.2, columnspacing=0.9, handletextpad=0.5)
        fig.subplots_adjust(left=0.17, right=0.97, bottom=0.15, top=0.72)
        return save(fig, out_stem)

    ax_curve.text(0.015, 0.03, "(a)", transform=ax_curve.transAxes, fontweight="bold")

    # --- bottom: test-set identification of predicted tokens ---
    style_axes(ax_bar, minor_x=False)
    evals = [(r, read_csv(src / f"eval{r[0]}.csv")[0]) for r in RUNS if (src / f"eval{r[0]}.csv").exists()]
    n = len(evals)
    width = 0.8 / max(n, 1)
    x = np.arange(len(METRICS))
    for i, ((_, label, line, face, _, _, hatch), row) in enumerate(evals):
        vals = [100 * float(row[key]) for key, _ in METRICS]
        ax_bar.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.92, color=face, edgecolor=line,
                   linewidth=0.6, hatch=hatch, zorder=3)
    ax_bar.set_xticks(x, [name for _, name in METRICS])
    ax_bar.set_ylabel("Test accuracy (%)")
    ax_bar.set_ylim(0, 100)
    ax_bar.text(0.985, 0.90, "(b)", transform=ax_bar.transAxes, fontweight="bold", ha="right")

    fig.legend(*ax_curve.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.54, 1.0),
               ncol=2, frameon=False, handlelength=2.2, columnspacing=0.9, handletextpad=0.5)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.07, top=0.84, hspace=0.42)
    save(fig, out_stem)


def save(fig, out_stem):
    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=600)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--curves-only"]
    default_out = "papers/AAAI_workshop_paper/Figures/loss_sweep" + ("_valcos" if "--curves-only" in sys.argv else "")
    main(args[0], args[1] if len(args) > 1 else default_out, curves_only="--curves-only" in sys.argv)
