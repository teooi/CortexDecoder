"""Released subject-1 reconstructions of MindEye2 (40 h) and NeuroPictor for the images in our comparison grid.

    modal run code/v2/scripts/baseline_recons.py --ids 3434,7859,...        (NSD 0-based image ids)

MindEye2: pscotti/mindeyev2 evals/final_subj01_pretrained_40sess_24bs/*_all_recons.pt (unrefined) and
*_all_enhancedrecons.pt (refined). The index -> image id order is checked against evals/all_images.pt and the NSD
stimuli (sorted shared-1000 ids). NeuroPictor: Fudan-fMRI/neuropictor results/sub01.zip. Downloads are cached on the
volume. Returns PNG bytes per method and id, plus a report of what was matched.
"""
import modal

DATA_DIR = "/mnt/braid2-flatmap-data"
CACHE = f"{DATA_DIR}/external_models/baseline_recons"
ME2 = "evals/final_subj01_pretrained_40sess_24bs/final_subj01_pretrained_40sess_24bs_{kind}.pt"
volume = modal.Volume.from_name("braid2-flatmap-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.8.0", "numpy", "scipy", "h5py", "pillow", "huggingface_hub")
app = modal.App("cortexdecoder-baseline-recons", image=image)


@app.function(volumes={DATA_DIR: volume}, cpu=4, memory=32768, timeout=60 * 60)
def fetch(ids: list[int]):
    import io
    import re
    import zipfile

    import h5py
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from scipy.io import loadmat

    def png(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    report, out = {}, {"mindeye2": {}, "mindeye2_refined": {}, "neuropictor": {}}
    shared = np.sort(loadmat(f"{DATA_DIR}/nsd_expdesign.mat")["sharedix"].reshape(-1).astype(np.int64) - 1)

    # --- MindEye2: check the index order with all_images.pt against the NSD stimuli ---
    get = lambda f: hf_hub_download("pscotti/mindeyev2", f, repo_type="dataset", cache_dir=CACHE)  # noqa: E731
    gt = torch.load(get("evals/all_images.pt"), map_location="cpu").float()  # [1000, 3, H, W] in [0, 1]
    with h5py.File(f"{DATA_DIR}/nsd_stimuli.hdf5", "r") as h5:
        probe = [0, 1, 500, 999]
        stims = {k: torch.from_numpy(h5["imgBrick"][int(shared[k])]).permute(2, 0, 1).float()[None] / 255 for k in probe}
    corr = {}
    for k in probe:
        a = torch.nn.functional.interpolate(stims[k], size=gt.shape[-2:], mode="bilinear").flatten()
        b = gt[k].flatten()
        corr[k] = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    report["mindeye2_order_check_corr_vs_sorted_shared"] = corr
    if min(corr.values()) < 0.95:
        raise RuntimeError(f"MindEye2 order is not sorted shared-1000: {corr}")
    row = {int(i): k for k, i in enumerate(shared)}
    for key, kind in [("mindeye2", "all_recons"), ("mindeye2_refined", "all_enhancedrecons")]:
        rec = torch.load(get(ME2.format(kind=kind)), map_location="cpu")
        report[f"{key}_shape"] = list(rec.shape)
        for i in ids:
            x = rec[row[i]].float().clamp(0, 1).permute(1, 2, 0).numpy()
            out[key][i] = png((x * 255).round().astype(np.uint8))
        del rec
    volume.commit()

    # --- NeuroPictor: find each id in the released zip ---
    z = zipfile.ZipFile(hf_hub_download("Fudan-fMRI/neuropictor", "results/sub01.zip", cache_dir=CACHE))
    names = [n for n in z.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    report["neuropictor_n_images"] = len(names)
    report["neuropictor_dirs"] = sorted({n.rsplit("/", 1)[0] for n in names})
    # Files are named by batch/index, not image id: match each ground-truth image to the shared-1000 stimuli by pixels.
    gt_names = sorted(n for n in names if "/gt/" in n)
    small = lambda im: torch.from_numpy(np.asarray(im.convert("RGB").resize((32, 32), Image.BILINEAR))).float().flatten()  # noqa: E731
    G = torch.stack([small(Image.open(io.BytesIO(z.read(n)))) for n in gt_names])
    with h5py.File(f"{DATA_DIR}/nsd_stimuli.hdf5", "r") as h5:
        T = torch.stack([small(Image.fromarray(h5["imgBrick"][int(i)])) for i in ids])
    zs = lambda x: (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True)  # noqa: E731
    sim = zs(T) @ zs(G).T / T.shape[1]  # [ids, gt]
    best, arg = sim.max(1)
    recon_dirs = [d for d in report["neuropictor_dirs"] if not d.endswith("/gt")]
    report["neuropictor_match"] = {}
    for i, s, k in zip(ids, best.tolist(), arg.tolist()):
        stem = gt_names[k].rsplit("/", 1)[-1]
        report["neuropictor_match"][i] = [gt_names[k], round(s, 4)]
        for d in recon_dirs:
            if f"{d}/{stem}" in names and s > 0.95:
                out.setdefault(f"neuropictor_{d.rsplit('/', 1)[-1]}", {})[i] = z.read(f"{d}/{stem}")
    return report, out


@app.local_entrypoint()
def main(ids: str, out_dir: str):
    import json
    import os
    report, out = fetch.remote([int(i) for i in ids.split(",")])
    print(json.dumps(report, indent=1))
    for method, images in out.items():
        os.makedirs(f"{out_dir}/{method}", exist_ok=True)
        for i, data in images.items():
            with open(f"{out_dir}/{method}/img{int(i):06d}.png", "wb") as f:
                f.write(data)
