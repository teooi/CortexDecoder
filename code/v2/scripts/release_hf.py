"""Publish the Subject 1 flatmaps, reconstructions and bridge weights to Hugging Face, straight from the Modal volume.

    modal deploy code/v2/scripts/release_hf.py
    python3 -c "import modal; print(modal.Function.from_name('cortexdecoder-hf-release', 'release').spawn(True).object_id)"   # dry run
    python3 -c "import modal; print(modal.Function.from_name('cortexdecoder-hf-release', 'release').spawn(False).object_id)"  # upload

The dry run builds the safetensors, zips and cards in /tmp, checks them, and prints what would be uploaded. The upload
creates both repos (the dataset is gated before any data goes up), skips files already in the repo, then compares every
uploaded file's size and SHA-256 with the source. Needs the Modal secret `huggingface` with a write token in HF_TOKEN.
"""
from pathlib import Path

import modal

DATA_DIR = "/mnt/braid2-flatmap-data"
DATASET_REPO = "teooi/cortexdecoder-nsd-flatmaps"
MODEL_REPO = "teooi/cortexdecoder"
FLATMAP_DIR = f"{DATA_DIR}/raw_flatmaps_cortexmae/fsaverage_fslr32k_cortexmae224x560_trainstats_v1_seed42_val0.05"
ROI_FILE = f"{DATA_DIR}/outputs/token_diagnostics/subj01_roi_overlap.json"
RUN = f"{DATA_DIR}/outputs/production_{{}}_subject01_baseline_shared1000_image_id_holdout_baseline_subj01_{{}}"
ROI_TAG = "tok-flatmap_tokens_roi-nsdgeneral_fp16"
FINAL_DIR = RUN.format("final_diffusion", f"20261030_232219_dec-16.0_{ROI_TAG}_v{ROI_TAG}")
EXPORTS = f"{FINAL_DIR}/drive_image_exports_singlehead_finaleval1000_s14_n38_no_sdxl/subj01"
CHECKPOINTS = {"vgg": f"{RUN.format('vgg', f'20261106_120000_{ROI_TAG}')}/vgg_best.pt",
               "clip": f"{RUN.format('clip', f'20261030_232219_dec-16.0_{ROI_TAG}')}/clip_best.pt"}
PAPER = "https://www.biorxiv.org/content/10.64898/2026.10.09.757766v1"
CODE = "https://github.com/teooi/CortexDecoder"
NSD_TERMS = "https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions"

volume = modal.Volume.from_name("braid2-flatmap-data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "numpy", "scipy", "safetensors", "huggingface_hub>=0.25")
         .add_local_dir(Path(__file__).parent.parent / "cortexdecoder", "/root/cortexdecoder"))
app = modal.App("cortexdecoder-hf-release", image=image)

CITATION = """```bibtex
@article{imototar2026cortexdecoder,
  title   = {CortexDecoder: fMRI-to-image Reconstruction from Cortical Flatmaps},
  author  = {Imoto-Tar, Teo and Fung, Maxwell and Gurha, Arnav},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.10.09.757766}
}
```"""


def dataset_card(shape, n_sessions):
    return f"""---
license: other
license_name: nsd-terms-and-conditions
license_link: {NSD_TERMS}
pretty_name: CortexDecoder NSD cortical flatmaps
tags: [fmri, neuroscience, brain-decoding, natural-scenes-dataset]
extra_gated_prompt: "This dataset is derived from the Natural Scenes Dataset (NSD). To access it, you agree to the NSD Terms and Conditions ({NSD_TERMS})."
extra_gated_fields:
  I agree to the NSD Terms and Conditions: checkbox
---

# CortexDecoder NSD cortical flatmaps

Cortical flatmaps and reconstructions for NSD Subject 1, from [CortexDecoder]({PAPER}). Code is on [GitHub]({CODE}) and the
model weights are in [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}).

## Files

```
flatmaps/subj01/
  subj01_sessionNN.npy               {shape} float32 flatmaps, one per trial ({n_sessions} sessions, session 11 excluded)
  subj01_sessionNN_image_ids.npy     NSD image index (0-based, 73k set) of each trial
  subj01_sessionNN.json              per-session record
  valid_mask.npy                     224x560 cortex mask
  manifest.json                      preprocessing record with the SHA-256 of every file
roi/subj01_roi_overlap.json          visual-cortex ROI on the CortexMAE patch grid
reconstructions/subj01/
  final.zip                          998 final reconstructions (PNG, 256 px, one diffusion sample per image)
  lowlevel.zip                       998 low-level Deep Image Prior reconstructions (PNG, 256 px)
  image_ids.csv                      file name and NSD image index of each reconstruction
```

## Flatmaps

NSD single-trial betas are resampled from fsaverage to fsLR-32k and projected onto CortexMAE's 224x560 flatmap. Each
flatmap location is standardized with statistics fitted on training images only (seed 42, 5% of non-test images held out
for validation). Pixels outside `valid_mask.npy` are not cortex.

The flatmaps cover the whole cortex. The visual-cortex ROI used by the final model is a mask over CortexMAE's 364 patches
(16x16 pixels each). `roi/subj01_roi_overlap.json` gives each patch's grid cell (`cells_row_col`) and the fraction of it
inside NSD's nsdgeneral mask (`coverage`). The ROI is the 50 patches with coverage of at least 0.5.

```python
import numpy as np
flatmaps = np.load("flatmaps/subj01/subj01_session01.npy")          # {shape}
image_ids = np.load("flatmaps/subj01/subj01_session01_image_ids.npy")
```

## Reconstructions

The 998 shared test images that Subject 1 viewed. Files are named `<row>_img<NSD image index>.png`. `final.zip` matches the
CortexDecoder row of Tables 1 and 2 in the paper, and `lowlevel.zip` the CortexDecoder (low-level) row.

## Citation

{CITATION}

Please also cite the Natural Scenes Dataset (Allen et al., 2022).
"""


def model_card(info):
    return f"""---
license: mit
library_name: pytorch
tags: [fmri, neuroscience, brain-decoding, natural-scenes-dataset]
datasets: [{DATASET_REPO}]
---

# CortexDecoder

Bridge weights for [CortexDecoder]({PAPER}), trained on NSD Subject 1. Code is on [GitHub]({CODE}) and the flatmaps and
reconstructions are in [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO}).

A frozen CortexMAE encoder turns visual-cortex flatmap patches into 200 brain tokens of dimension 768. Two cross-attention
bridges map those tokens to image features.

| File | Bridge | Output | Parameters |
|---|---|---|---|
| `vgg_bridge.safetensors` | low-level | 7190 VGG16-BN tokens of dimension 512, inverted to an image with a Deep Image Prior | {info['vgg']}M |
| `clip_bridge.safetensors` | semantic | 256 OpenCLIP ViT-bigG/14 tokens of dimension 1664, used to condition SDXL-unCLIP | {info['clip']}M |

`checkpoints/` has the original training checkpoints, which also contain the optimizer state. `config.json` lists the
bridge settings.

## Loading

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from cortexdecoder.core.config import Config                     # from the GitHub repo, code/v2
from cortexdecoder.models.bridge import FlatmapSingleHeadBridge

bridge = FlatmapSingleHeadBridge(Config(), brain_dim=768, num_patches=200, target="clip")
bridge.load_state_dict(load_file(hf_hub_download("{MODEL_REPO}", "clip_bridge.safetensors")))
clip_tokens = bridge(brain_tokens)                               # [B, 200, 768] -> [B, 256, 1664]
```

The brain tokens come from `code/v2/scripts/generate_roi_tokens.py`, which runs CortexMAE with only the visual-cortex
patches visible.

## Results

NSD Subject 1, 998 test images, one diffusion sample per image.

| PixCorr | SSIM | Alex(2) | Alex(5) | Incep | CLIP | Eff | SwAV |
|---|---|---|---|---|---|---|---|
| 0.366 | 0.413 | 98.1% | 99.1% | 94.9% | 93.5% | 0.628 | 0.360 |

## Citation

{CITATION}
"""


@app.function(volumes={DATA_DIR: volume}, secrets=[modal.Secret.from_name("huggingface")], cpu=4, memory=16384,
              timeout=4 * 60 * 60)
def release(dry_run: bool = True):
    import csv
    import hashlib
    import json
    import os
    import re
    import zipfile

    import numpy as np
    import torch
    from huggingface_hub import CommitOperationAdd, HfApi
    from safetensors.torch import load_file, save_file
    from scipy.io import loadmat

    from cortexdecoder.core.config import Config
    from cortexdecoder.models.bridge import FlatmapSingleHeadBridge

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(f"no HF_TOKEN in the 'huggingface' secret; keys present: {sorted(k for k in os.environ if 'HF' in k.upper() or 'HUGG' in k.upper())}")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    print("HF user:", user)
    if user != DATASET_REPO.split("/")[0]:
        raise RuntimeError(f"token belongs to {user!r}, but the repos are under {DATASET_REPO.split('/')[0]!r}")

    tmp = Path("/tmp/release")
    tmp.mkdir(exist_ok=True)

    # --- dataset files --------------------------------------------------------
    dataset = {}  # path in repo -> local path
    sessions = sorted(int(m.group(1)) for f in os.listdir(FLATMAP_DIR) if (m := re.fullmatch(r"subj01_session(\d+)\.npy", f)))
    for s in sessions:
        for suffix in (".npy", "_image_ids.npy", ".json"):
            name = f"subj01_session{s:02d}{suffix}"
            dataset[f"flatmaps/subj01/{name}"] = f"{FLATMAP_DIR}/{name}"
    for name in ("valid_mask.npy", "manifest.json"):
        dataset[f"flatmaps/subj01/{name}"] = f"{FLATMAP_DIR}/{name}"
    dataset["roi/subj01_roi_overlap.json"] = ROI_FILE

    shape = tuple(np.load(f"{FLATMAP_DIR}/subj01_session01.npy", mmap_mode="r").shape)
    # Image ids must be the 0-based NSD indices the pipeline uses (data/nsd.py).
    mat = loadmat(f"{DATA_DIR}/nsd_expdesign.mat")
    expected = (mat["subjectim"].astype(np.int64) - 1)[0, mat["masterordering"].reshape(-1).astype(np.int64) - 1]
    for s in sessions:
        ids = np.load(f"{FLATMAP_DIR}/subj01_session{s:02d}_image_ids.npy")
        assert np.array_equal(ids, expected[(s - 1) * 750:(s - 1) * 750 + len(ids)]), f"session {s} image ids differ from NSD"
    roi = json.load(open(ROI_FILE))
    n_roi = sum(c >= 0.5 for c in roi["coverage"])
    assert len(roi["coverage"]) == 364 and n_roi == 50, (len(roi["coverage"]), n_roi)
    print(f"flatmaps: {len(sessions)} sessions, shape {shape}, image ids match NSD; ROI: {n_roi} of 364 patches")

    rows = []
    for name, folder in [("final", "final_diffusion"), ("lowlevel", "vgg_dip_preview")]:
        files = sorted(os.listdir(f"{EXPORTS}/{folder}"))
        assert len(files) == 998 and all(f.endswith(".png") for f in files), (folder, len(files))
        path = tmp / f"{name}.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:  # PNGs are already compressed
            for f in files:
                z.write(f"{EXPORTS}/{folder}/{f}", f)
        dataset[f"reconstructions/subj01/{name}.zip"] = str(path)
        if not rows:
            rows = [(f, int(f.split("_img")[1][:-4])) for f in files]
    with open(tmp / "image_ids.csv", "w", newline="") as f:
        csv.writer(f).writerows([("file", "nsd_image_index"), *rows])
    dataset["reconstructions/subj01/image_ids.csv"] = str(tmp / "image_ids.csv")
    (tmp / "dataset_README.md").write_text(dataset_card(shape, len(sessions)))
    dataset["README.md"] = str(tmp / "dataset_README.md")

    # --- model files ----------------------------------------------------------
    model, params, config = {}, {}, {}
    for head, path in CHECKPOINTS.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        weights = {k: v.contiguous() for k, v in ckpt["model"].items()}
        out = tmp / f"{head}_bridge.safetensors"
        save_file(weights, str(out))
        cfg = Config(n_layers=ckpt["n_layers"], n_heads=ckpt["n_heads"], d_model=ckpt["d_model"])
        bridge = FlatmapSingleHeadBridge(cfg, brain_dim=768, num_patches=200, target=head)
        bridge.load_state_dict(load_file(str(out)))  # strict: same keys and shapes as the training code builds
        params[head] = round(sum(p.numel() for p in bridge.parameters()) / 1e6, 1)
        config[f"{head}_bridge"] = {"d_model": cfg.d_model, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                                    "ffn_dim": cfg.ffn_dim, "dropout": cfg.dropout, "brain_dim": 768, "num_brain_tokens": 200,
                                    "brain_tokens": ckpt.get("brain_tokens", "flatmap_tokens_roi-nsdgeneral_fp16"),
                                    "best_epoch": int(ckpt["epoch"]), "parameters_millions": params[head]}
        if head == "clip":
            config["clip_bridge"].update(clip_loss=ckpt["clip_loss"], decoder_loss_weight=ckpt["decoder_loss_weight"])
        model[f"{head}_bridge.safetensors"] = str(out)
        model[f"checkpoints/{head}_best.pt"] = path
    print("bridge parameters (M):", params)
    (tmp / "config.json").write_text(json.dumps(config, indent=2))
    model["config.json"] = str(tmp / "config.json")
    (tmp / "model_README.md").write_text(model_card(params))
    model["README.md"] = str(tmp / "model_README.md")

    for repo, files in [(DATASET_REPO, dataset), (MODEL_REPO, model)]:
        total = sum(os.path.getsize(p) for p in files.values())
        print(f"\n{repo}: {len(files)} files, {total / 1e9:.2f} GB")
        for k, p in sorted(files.items()):
            if not k.startswith("flatmaps/subj01/subj01_session") or k.endswith("session01.npy"):
                print(f"  {os.path.getsize(p) / 1e6:10.1f} MB  {k}")
    if dry_run:
        print("\ndry run: nothing uploaded")
        return {"dataset_files": len(dataset), "model_files": len(model), "params": params, "shape": shape}

    # --- upload ---------------------------------------------------------------
    api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
    api.update_repo_settings(DATASET_REPO, repo_type="dataset", gated="auto")  # gate before any data goes up
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)

    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()

    report = {}
    for repo, repo_type, files in [(MODEL_REPO, "model", model), (DATASET_REPO, "dataset", dataset)]:
        have = {f.path: f for f in api.list_repo_tree(repo, repo_type=repo_type, recursive=True, expand=True)
                if hasattr(f, "size")}
        todo = [k for k, p in files.items()
                if k not in have or have[k].size != os.path.getsize(p) or k.endswith(".md")]
        print(f"{repo}: uploading {len(todo)} of {len(files)} files")
        batch = []
        for k in sorted(todo):
            batch.append(k)
            if sum(os.path.getsize(files[b]) for b in batch) > 2e9 or k == sorted(todo)[-1]:
                api.create_commit(repo, repo_type=repo_type, commit_message=f"Add {len(batch)} files",
                                  operations=[CommitOperationAdd(b, files[b]) for b in batch])
                print("  committed", batch[0], "...", batch[-1])
                batch = []

        # Verify every file against the source: size, and SHA-256 for LFS files.
        remote = {f.path: f for f in api.list_repo_tree(repo, repo_type=repo_type, recursive=True, expand=True)
                  if hasattr(f, "size")}
        bad = []
        for k, p in files.items():
            r = remote.get(k)
            if r is None or r.size != os.path.getsize(p):
                bad.append(k)
            elif r.lfs is not None and r.lfs.sha256 != sha256(p):
                bad.append(k)
        report[repo] = {"files": len(files), "mismatched": bad}
        print(f"{repo}: verified {len(files) - len(bad)} of {len(files)} files", "mismatched: " + str(bad) if bad else "")
    info = api.dataset_info(DATASET_REPO)
    report["dataset_gated"] = info.gated
    print("dataset gated:", info.gated)
    return report


@app.local_entrypoint()
def main(upload: bool = False):
    print(release.remote(not upload))
