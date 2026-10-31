"""Bridge-size ablation figure: validation token cosine per epoch, same paper style as plot_loss_sweep.py.

    .venv/bin/python code/v2/scripts/plot_size_sweep.py <dir with hist_*.csv> [out_stem]

Inputs are each run's train_clip_history.csv renamed hist_<run>.csv (hist_base, hist_L-2, hist_H-16, hist_L-8, hist_L-12).
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_loss_sweep import save, style_axes  # shared rcParams (9 pt Times, TrueType fonts), axes style, saving

# Depth: shades of the diagrams' semantic-feature blue (darker = deeper). Heads variant: query-token orange, dashed.
RUNS = [
    # (file stem, label, line colour, face colour, linestyle, marker)
    ("hist_L-2", "2 blocks, 8 heads (20.4M)", "#7FB0DC", "#D6E8F7", "-", "o"),
    ("hist_base", "4 blocks, 8 heads (39.3M)", "#2F6FA8", "#AED2EF", "-", "o"),
    ("hist_H-16", "4 blocks, 16 heads (39.3M)", "#C9741F", "#F9C784", "--", "^"),
    ("hist_L-8", "8 blocks, 8 heads (77.0M)", "#1E4D78", "#7FB0DC", "-", "s"),
    ("hist_L-12", "12 blocks, 8 heads (114.6M)", "#0F2C47", "#5B8DBA", (0, (4, 1.5, 1, 1.5)), "D"),
]


def main(src, out_stem):
    src = Path(src)
    fig, ax = plt.subplots(figsize=(3.3, 3.8))
    style_axes(ax)
    for stem, label, line, face, style, marker in RUNS:
        path = src / f"{stem}.csv"
        if not path.exists():
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        epochs = [int(float(r["epoch"])) for r in rows]
        val_cos = [float(r["val_cos"]) for r in rows]
        ax.plot(epochs, val_cos, label=label, color=line, linestyle=style, linewidth=1.1,
                marker=marker, markersize=3.0, markevery=2,
                markerfacecolor=face, markeredgecolor=line, markeredgewidth=0.6, zorder=3)
        best = int(np.argmax(val_cos))
        ax.plot(epochs[best], val_cos[best], marker="*", markersize=6.5, color=line,
                markeredgecolor="black", markeredgewidth=0.4, linestyle="none", zorder=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val. token cosine")
    ax.set_xlim(0.5, 25.5)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.55, 1.0),
               ncol=1, frameon=False, handlelength=2.2, handletextpad=0.5, labelspacing=0.25)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.10, top=0.76)
    save(fig, out_stem)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "papers/AAAI_workshop_paper/Figures/size_sweep_valcos")
