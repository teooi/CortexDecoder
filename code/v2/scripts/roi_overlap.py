"""Are the most repeat-reliable CortexMAE patches the nsdgeneral visual-cortex ROI?

    modal run code/v2/scripts/roi_overlap.py

Projects NSD's fsaverage nsdgeneral labels to fsLR-32k (wb_command ADAP_BARY_AREA, as the betas were) and onto the
CortexMAE 224x560 flatmap with cortex_mae's own resampler, then computes each of the 364 in-brain 16x16 patches'
nsdgeneral coverage. CortexMAE keeps in-brain patches in row-major grid order (masking.trim_patch_mask without a mask
ratio), so patch p is the p-th in-brain cell. Compares coverage with the reliability ranking from diagnose_tokens.py.
Writes <data>/outputs/token_diagnostics/subj01_roi_overlap.{json,pdf,png}.
"""
import modal

DATA_DIR = "/mnt/braid2-flatmap-data"
LABEL_URL = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata/freesurfer/fsaverage/label/{h}.nsdgeneral.mgz"
ATLAS = {"current_sphere": "fsaverage_std_sphere.{H}.164k_fsavg_{H}.surf.gii",
         "current_area": "fsaverage.{H}.midthickness_va_avg.164k_fsavg_{H}.shape.gii",
         "new_sphere": "fs_LR-deformed_to-fsaverage.{H}.sphere.32k_fs_LR.surf.gii",
         "new_area": "fs_LR.{H}.midthickness_va_avg.32k_fs_LR.shape.gii"}
PATCH, GRID_H, GRID_W = 16, 14, 35

volume = modal.Volume.from_name("braid2-flatmap-data")
image = (modal.Image.debian_slim(python_version="3.11").apt_install("wget", "connectome-workbench", "build-essential")
         .pip_install("torch==2.8.0", "cortex_mae==0.1.1", "nibabel", "numpy", "scipy", "matplotlib"))
app = modal.App("cortexdecoder-roi-overlap", image=image)


@app.function(volumes={DATA_DIR: volume}, cpu=4, memory=16384, timeout=30 * 60)
def overlap():
    import json
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import torch
    from cortex_mae import nisc
    from scipy.stats import spearmanr

    tmp, out_dir = "/tmp/roi", f"{DATA_DIR}/outputs/token_diagnostics"
    os.makedirs(tmp, exist_ok=True)
    hemis = []
    for h, H in [("lh", "L"), ("rh", "R")]:
        mgh = f"{tmp}/{h}.mgz"
        assert os.system(f"wget -q {LABEL_URL.format(h=h)} -O {mgh}") == 0
        roi = (nib.load(mgh).get_fdata().reshape(-1) > 0).astype(np.float32)  # nsdgeneral: 1 inside, <=0 outside
        raw, res = f"{tmp}/{H}_raw.func.gii", f"{tmp}/{H}_32k.func.gii"
        nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(roi)]), raw)
        a = {k: f"{DATA_DIR}/hcp_standard_mesh_atlases/{v.format(H=H)}" for k, v in ATLAS.items()}
        assert os.system(f"wb_command -metric-resample {raw} {a['current_sphere']} {a['new_sphere']} "
                         f"ADAP_BARY_AREA {res} -area-metrics {a['current_area']} {a['new_area']}") == 0
        hemis.append(nib.load(res).darrays[0].data)
    fslr = np.concatenate(hemis)[None].astype(np.float32)  # [1, 64984], fraction of each vertex inside nsdgeneral

    resampler = nisc.flat_resampler_fslr64k_224_560()
    roi_img = resampler.transform(fslr, interpolation="linear")[0]  # [224, 560]
    brain = resampler.mask_
    cells = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            sl = (slice(r * PATCH, (r + 1) * PATCH), slice(c * PATCH, (c + 1) * PATCH))
            if brain[sl].any():
                cells.append((r, c, float(roi_img[sl][brain[sl]].mean())))
    assert len(cells) == 364, f"expected 364 in-brain patches, got {len(cells)}"
    coverage = np.array([c[2] for c in cells])

    diag = torch.load(f"{out_dir}/subj01_token_diagnostics.pt")
    rel = diag["reliability_eval_half"].numpy()  # held-out half; the ranking used the other half
    ranked = diag["ranked_patches"].numpy()
    in_roi = coverage >= 0.5
    result = {"n_patches": 364, "n_nsdgeneral_patches_cov50": int(in_roi.sum()),
              "spearman_reliability_vs_coverage": float(spearmanr(rel, coverage).correlation)}
    for k in (16, 32, 64, 128):
        top = ranked[:k]
        result[f"top{k}"] = {"mean_coverage": float(coverage[top].mean()),
                             "frac_in_nsdgeneral_cov50": float(in_roi[top].mean())}
    k = int(in_roi.sum())
    top_k = set(ranked[:k].tolist())
    roi_set = set(np.where(in_roi)[0].tolist())
    result["topK_vs_nsdgeneral_jaccard_K=n_roi"] = len(top_k & roi_set) / len(top_k | roi_set)
    result["mean_reliability_inside_vs_outside"] = [float(rel[in_roi].mean()), float(rel[~in_roi].mean())]

    # Figure: reliability map and nsdgeneral coverage on the 14x35 patch grid, top-64 outlined.
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    grids = {name: np.full((GRID_H, GRID_W), np.nan) for name in ("rel", "cov")}
    for p, (r, c, cov) in enumerate(cells):
        grids["rel"][r, c], grids["cov"][r, c] = rel[p], cov
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.9), constrained_layout=True)
    for ax, key, title, cmap in [(axes[0], "rel", "Patch repeat reliability", "magma"),
                                 (axes[1], "cov", "nsdgeneral coverage", "Blues")]:
        im = ax.imshow(grids[key], cmap=cmap, interpolation="nearest")
        for p in ranked[:64]:
            r, c, _ = cells[p]
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="#2CA02C", linewidth=0.8))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(f"{out_dir}/subj01_roi_overlap.pdf")
    fig.savefig(f"{out_dir}/subj01_roi_overlap.png", dpi=600)
    with open(f"{out_dir}/subj01_roi_overlap.json", "w") as f:
        json.dump({**result, "coverage": coverage.tolist(), "cells_row_col": [(r, c) for r, c, _ in cells]}, f, indent=2)
    volume.commit()
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main():
    print(overlap.remote())
