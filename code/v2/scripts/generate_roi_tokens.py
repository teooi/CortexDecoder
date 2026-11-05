"""CortexMAE brain tokens with an encoder-level nsdgeneral ROI: only the ROI patches are visible to the encoder.

    modal deploy code/v2/scripts/generate_roi_tokens.py
    python3 -c "import modal; print(modal.Function.from_name('cortexdecoder-roi-tokens', 'generate').spawn('subj01').object_id)"

Same pipeline as BRAID2/modal_jobs/generate_flatmap_tokens.py (NSD fsaverage betas -> wb_command ADAP_BARY_AREA ->
fsLR-32k -> 16-frame CIFTI -> CortexMAE flatmap transform), but instead of run_embedding() the encoder gets CortexMAE's
own validity mask intersected with the ROI patches (scripts/roi_overlap.py, coverage >= ROI_MIN_COVERAGE). Unobserved
patches are dropped before the transformer, exactly as background (and pretraining-masked) patches are, so the tokens
depend on nsdgeneral activity only. Output per session: [750, 4 * n_roi, 768] fp16, time-major like the full tokens.

Before the first session it re-encodes one chunk without the ROI and compares it with the production whole-cortex tokens,
so a pipeline or package mismatch stops the run before it costs anything.
"""
import json
import os
import time

import modal

DATA_DIR = "/mnt/braid2-flatmap-data"
OUT_DIR = f"{DATA_DIR}/flatmap_tokens_roi-nsdgeneral_fp16"
REF_DIR = f"{DATA_DIR}/flatmap_tokens_fp16"
ROI_FILE = f"{DATA_DIR}/outputs/token_diagnostics/subj01_roi_overlap.json"
ROI_MIN_COVERAGE = 0.5  # Config.roi_min_coverage
PATCH = 16
KNOWN_BAD_SESSIONS = {("subj01", 11)}
MAX_SESSIONS = 40
FSAVG_BETA_URL = ("https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/"
                  "{subject}/fsaverage/betas_fithrf_GLMdenoise_RR/{hemi}.betas_session{sess:02d}.mgh")
ATLAS_DIR = f"{DATA_DIR}/hcp_standard_mesh_atlases"  # cached by generate_flatmap_tokens.py
ATLAS = {"current_sphere": "fsaverage_std_sphere.{H}.164k_fsavg_{H}.surf.gii",
         "current_area": "fsaverage.{H}.midthickness_va_avg.164k_fsavg_{H}.shape.gii",
         "new_sphere": "fs_LR-deformed_to-fsaverage.{H}.sphere.32k_fs_LR.surf.gii",
         "new_area": "fs_LR.{H}.midthickness_va_avg.32k_fs_LR.shape.gii"}
FSAVERAGE_N_VERTICES = 163842
NUM_FRAMES = 16
TRIALS_PER_CIFTI = 50

volume = modal.Volume.from_name("braid2-flatmap-data")
image = (modal.Image.debian_slim(python_version="3.11").apt_install("wget", "connectome-workbench")
         .pip_install("torch==2.8.0", "cortex_mae==0.1.1", "nibabel", "numpy", "einops", "tqdm"))
app = modal.App("cortexdecoder-roi-tokens", image=image)


@app.function(gpu="A10G", cpu=4, memory=16384, volumes={DATA_DIR: volume}, timeout=8 * 60 * 60)
def generate(subject: str = "subj01"):
    import nibabel as nib
    import numpy as np
    import torch
    from cortex_mae import CortexMAE
    from cortex_mae.inference import pad_unfold
    from cortex_mae.nisc import FSLR64K_NUM_VERTICES, fetch_schaefer, get_brain_model_axis

    tmp = "/tmp/roi_tokens"
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(ROI_FILE) as f:
        roi = json.load(f)
    roi_cells = [rc for rc, cov in zip(roi["cells_row_col"], roi["coverage"]) if cov >= ROI_MIN_COVERAGE]
    roi_pix = torch.zeros(224, 560)
    for r, c in roi_cells:
        roi_pix[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = 1
    print(f"ROI: {len(roi_cells)} patches (coverage >= {ROI_MIN_COVERAGE})")

    mae = CortexMAE.from_pretrained("cortex_mae_flat")
    device = mae.get_device()
    axis = get_brain_model_axis(nib.load(fetch_schaefer(400, space="fslr64k")))

    def resample(sess):
        out = {}
        for hemi, H in [("lh", "L"), ("rh", "R")]:
            mgh = f"{tmp}/{hemi}.mgh"
            url = FSAVG_BETA_URL.format(subject=subject, hemi=hemi, sess=sess)
            if os.system(f"wget -q {url} -O {mgh}") != 0:
                return None
            data = nib.load(mgh).get_fdata().reshape(FSAVERAGE_N_VERTICES, -1)
            raw, fslr = f"{tmp}/{H}_raw.func.gii", f"{tmp}/{H}_32k.func.gii"
            nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(np.ascontiguousarray(data[:, i], np.float32))
                                                   for i in range(data.shape[1])]), raw)
            a = {k: f"{ATLAS_DIR}/{v.format(H=H)}" for k, v in ATLAS.items()}
            cmd = (f"wb_command -metric-resample {raw} {a['current_sphere']} {a['new_sphere']} ADAP_BARY_AREA {fslr} "
                   f"-area-metrics {a['current_area']} {a['new_area']}")
            if os.system(cmd) != 0:
                raise RuntimeError(f"wb_command failed: {cmd}")
            out[H] = np.stack([d.data for d in nib.load(fslr).darrays], axis=1)
            for p in (mgh, raw, fslr):
                os.remove(p)
        return out

    def embed(l_chunk, r_chunk, use_roi=True):
        """[n_trials, 4 * n_visible_patches, 768] for one chunk, mirroring CortexMAE.run_embedding."""
        rows = np.concatenate([l_chunk, r_chunk], axis=0).T.astype(np.float32)
        assert rows.shape[1] == FSLR64K_NUM_VERTICES, rows.shape
        series = np.repeat(rows, NUM_FRAMES, axis=0)
        cifti = f"{tmp}/chunk.dtseries.nii"
        header = nib.cifti2.Cifti2Header.from_axes((nib.cifti2.SeriesAxis(0.0, 1.0, series.shape[0]), axis))
        nib.save(nib.cifti2.Cifti2Image(series, header=header), cifti)
        bold, mask = mae._unpack_sample(mae.transform(mae.load_input(cifti)))
        os.remove(cifti)
        bold, mask, _ = pad_unfold(bold, mask, mae.num_frames)
        if use_roi:
            mask = mask * roi_pix
        out = []
        with torch.inference_mode(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            for b0 in range(0, len(bold), 8):
                _, _, patch = mae.model.encoder.forward_embedding(bold[b0:b0 + 8].to(device), mask[b0:b0 + 8].to(device))
                out.append(patch.float().cpu())
        return torch.cat(out)

    checked = False
    for sess in range(1, MAX_SESSIONS + 1):
        if (subject, sess) in KNOWN_BAD_SESSIONS:
            continue
        path = f"{OUT_DIR}/{subject}_session{sess:02d}_fp16.pt"
        if os.path.exists(path):
            continue
        t0 = time.time()
        res = resample(sess)
        if res is None:
            print(f"stopping at session {sess:02d} (betas not available)")
            break
        n = res["L"].shape[1]
        if not checked:
            ref = torch.load(f"{REF_DIR}/{subject}_session{sess:02d}_fp16.pt", map_location="cpu")[:TRIALS_PER_CIFTI].float()
            full = embed(res["L"][:, :TRIALS_PER_CIFTI], res["R"][:, :TRIALS_PER_CIFTI], use_roi=False)
            cos = torch.nn.functional.cosine_similarity(full.flatten(1), ref.flatten(1)).min().item()
            print(f"reproduction check vs production tokens: shapes {tuple(full.shape)} / {tuple(ref.shape)}, min cos {cos:.5f}")
            if full.shape != ref.shape or cos < 0.99:
                raise RuntimeError("pipeline does not reproduce the production whole-cortex tokens; not generating")
            checked = True
        tokens = torch.cat([embed(res["L"][:, s:s + TRIALS_PER_CIFTI], res["R"][:, s:s + TRIALS_PER_CIFTI])
                            for s in range(0, n, TRIALS_PER_CIFTI)])
        assert tokens.shape == (n, 4 * len(roi_cells), 768), tokens.shape
        if not torch.isfinite(tokens).all():
            print(f"session {sess:02d}: non-finite tokens, not saving")
            continue
        torch.save(tokens.half(), path)
        volume.commit()
        print(f"session {sess:02d} -> {path} {tuple(tokens.shape)} in {time.time() - t0:.0f}s")
    print("done")
