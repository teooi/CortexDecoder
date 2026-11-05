"""nsdgeneral ROI on the CortexMAE patch grid, styled like the other paper figures.

    .venv/bin/python code/v2/scripts/plot_roi_mask.py <subj01_roi_overlap.json> [out_stem]

The JSON is scripts/roi_overlap.py's output (per-patch nsdgeneral coverage + grid cell of each of the 364 patches).
Left: every cortical patch the whole-cortex bridge reads. Middle: nsdgeneral coverage, outlined where it reaches
roi_min_coverage. Right: the patches the ROI bridge keeps (time copies averaged, so one token per patch).
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from plot_loss_sweep import save  # shared rcParams (9 pt Times, TrueType fonts) and saving

GRID_H, GRID_W = 14, 35
MIN_COVERAGE = 0.5  # Config.roi_min_coverage
TEMPORAL = 4  # CortexMAE time chunks per patch
BLUE, LIGHT, EDGE = "#2F6FA8", "#E6E6E6", "#1E4D78"


def grid(values, cells):
    out = np.full((GRID_H, GRID_W), np.nan)
    for v, (r, c) in zip(values, cells):
        out[r, c] = v
    return out


def outline(ax, keep):
    """Draw the boundary of the kept cells."""
    for r, c in zip(*np.nonzero(keep)):
        for dr, dc, seg in [(-1, 0, ([c - .5, c + .5], [r - .5, r - .5])), (1, 0, ([c - .5, c + .5], [r + .5, r + .5])),
                            (0, -1, ([c - .5, c - .5], [r - .5, r + .5])), (0, 1, ([c + .5, c + .5], [r - .5, r + .5]))]:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < GRID_H and 0 <= cc < GRID_W and keep[rr, cc]):
                ax.plot(*seg, color="black", linewidth=0.9, solid_capstyle="round")


def panel(ax, title):
    ax.set_title(title, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main(src, out_stem):
    with open(src) as f:
        d = json.load(f)
    cells, coverage = d["cells_row_col"], np.asarray(d["coverage"])
    n, n_roi = len(cells), int((coverage >= MIN_COVERAGE).sum())
    cortex = grid(np.ones(n), cells)
    cov = grid(coverage, cells)
    keep = np.nan_to_num(cov) >= MIN_COVERAGE

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.55))
    grid_kw = dict(edgecolors="white", linewidth=0.3)

    axes[0].pcolormesh(np.arange(GRID_W + 1) - .5, np.arange(GRID_H + 1) - .5, cortex,
                       cmap=ListedColormap([BLUE]), **grid_kw)
    panel(axes[0], f"Whole cortex: {n} patches\n({n} × {TEMPORAL} = {n * TEMPORAL} tokens)")

    mesh = axes[1].pcolormesh(np.arange(GRID_W + 1) - .5, np.arange(GRID_H + 1) - .5, cov, cmap="Blues",
                              vmin=0, vmax=1, **grid_kw)
    outline(axes[1], keep)
    panel(axes[1], f"nsdgeneral coverage\n(outline: ≥ {MIN_COVERAGE:g})")
    cbar = fig.colorbar(mesh, ax=axes[1], fraction=0.035, pad=0.02, ticks=[0, 0.5, 1])
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(width=0.4, length=2)

    roi = np.where(np.isnan(cortex), np.nan, keep.astype(float))
    axes[2].pcolormesh(np.arange(GRID_W + 1) - .5, np.arange(GRID_H + 1) - .5, roi,
                       cmap=ListedColormap([LIGHT, BLUE]), vmin=0, vmax=1, **grid_kw)
    outline(axes[2], keep)
    panel(axes[2], f"ROI bridge input: {n_roi} patches\n(time-averaged, {n_roi} tokens)")

    for ax in axes:
        ax.set_xlim(-.5, GRID_W - .5)
        ax.set_ylim(GRID_H - .5, -.5)
        ax.set_aspect("equal")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.07, top=0.80, wspace=0.12)
    save(fig, out_stem)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "papers/AAAI_workshop_paper/Figures/roi_mask")
