"""Compute SigLIP 2 patch-token targets for every NSD image a subject saw, matching the SigLIP2 unCLIP training recipe.

    modal run --detach code/v2/scripts/extract_siglip2_targets.py            # subject 1
    modal run --detach code/v2/scripts/extract_siglip2_targets.py --subject 2

Mirrors code/extra/siglip2 finetune/brainit-siglip2-unclip-train.py exactly: same pinned encoder revision and package
versions, PIL bicubic short-side resize to 256 + center crop, (x - 0.5) / 0.5 in fp32 -> bf16, last_hidden_state under
bf16 autocast, stored as native bf16. NSD stimuli (425x425 squares the subject actually saw) are encoded, not COCO JPEGs.

Writes <data>/siglip2_targets_cache/subjNN_siglip2_targets.pt  {nsd image id (0-based): bf16 [256, 1536]}
       <data>/siglip2_targets_cache/subjNN_siglip2_targets.json (provenance)
"""
import modal

DATA_DIR = "/mnt/braid2-flatmap-data"
MODEL = "google/siglip2-giant-opt-patch16-256"
REVISION = "46d7129e1aa1527cc5a44d86fb35250df3abf0aa"  # = image_encoder_revision of production_siglip2_unclip_v4
SIZE, TOKENS, DIM, BATCH = 256, 256, 1536, 8  # batch 8 as in the unCLIP frozen-feature cache

volume = modal.Volume.from_name("braid2-flatmap-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.8.0", "torchvision==0.23.0", "transformers==4.51.3", "huggingface-hub==0.30.2",
    "safetensors==0.5.3", "numpy==2.2.6", "pillow==11.2.1", "h5py", "scipy", "scikit-image==0.25.2",
)
app = modal.App("cortexdecoder-siglip2-targets", image=image)


def preprocess(array, size=SIZE):
    """brainit-siglip2-unclip-train.py ImageDataset.__getitem__, applied to an in-memory NSD stimulus."""
    import numpy as np
    import torch
    from PIL import Image

    image = Image.fromarray(array).convert("RGB")
    w, h = image.size
    scale = size / min(w, h)
    rw, rh = max(size, round(w * scale)), max(size, round(h * scale))
    image = image.resize((rw, rh), Image.Resampling.BICUBIC)
    left, top = (rw - size) // 2, (rh - size) // 2
    image = image.crop((left, top, left + size, top + size))
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255


@app.function(gpu="H100", volumes={DATA_DIR: volume}, memory=65536, timeout=2 * 60 * 60)
def extract(subject: int = 1):
    import hashlib
    import json
    import os

    import h5py
    import numpy as np
    import torch
    from scipy.io import loadmat
    from tqdm.auto import tqdm
    from transformers import AutoImageProcessor, SiglipVisionModel

    hf_cache = f"{DATA_DIR}/external_models/huggingface"
    encoder = SiglipVisionModel.from_pretrained(MODEL, revision=REVISION, cache_dir=hf_cache,
                                                torch_dtype=torch.bfloat16).eval().requires_grad_(False).cuda()
    processor = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION, cache_dir=hf_cache, use_fast=False)
    assert encoder.config.image_size == SIZE and encoder.config.hidden_size == DIM
    assert (SIZE // encoder.config.patch_size) ** 2 == TOKENS
    mean = torch.tensor(processor.image_mean, device="cuda")[None, :, None, None]
    std = torch.tensor(processor.image_std, device="cuda")[None, :, None, None]

    @torch.no_grad()
    def encode(pixels):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = encoder(pixel_values=((pixels - mean) / std).to(torch.bfloat16)).last_hidden_state
        if tokens.shape[1:] != (TOKENS, DIM) or not torch.isfinite(tokens).all():
            raise RuntimeError("Invalid SigLIP patch tokens")
        return tokens.cpu()

    subjectim = loadmat(f"{DATA_DIR}/nsd_expdesign.mat")["subjectim"].astype(np.int64) - 1
    ids = sorted({int(i) for i in subjectim[subject - 1]})
    print(f"subj{subject:02d}: {len(ids)} unique images")

    stimuli = h5py.File(f"{DATA_DIR}/nsd_stimuli.hdf5", "r")["imgBrick"]
    targets = {}
    for start in tqdm(range(0, len(ids), BATCH), desc="SigLIP2 tokens", mininterval=10):
        batch_ids = ids[start:start + BATCH]
        pixels = torch.stack([preprocess(stimuli[i]) for i in batch_ids]).cuda()
        actual = len(batch_ids)
        if actual < BATCH:  # pad like the training cache so every call sees batch 8
            pixels = torch.cat([pixels, pixels[:1].expand(BATCH - actual, -1, -1, -1)])
        for img_id, tok in zip(batch_ids, encode(pixels)[:actual]):
            targets[img_id] = tok.clone()

    # Determinism check: re-encode the first batch and require bit-identical tokens.
    first = torch.stack([preprocess(stimuli[i]) for i in ids[:BATCH]]).cuda()
    again = encode(first)
    assert all(torch.equal(again[j], targets[ids[j]]) for j in range(BATCH)), "non-deterministic encoding"

    out_dir = f"{DATA_DIR}/siglip2_targets_cache"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/subj{subject:02d}_siglip2_targets.pt"
    torch.save(targets, out)
    stacked = torch.stack([targets[i] for i in ids[:256]]).float()
    meta = {
        "subject": subject, "count": len(targets), "model": MODEL, "revision": REVISION,
        "token_shape": [TOKENS, DIM], "dtype": "bfloat16",
        "token_source": "native_post_layernorm_patch_tokens_no_pooling_no_l2",
        "preprocess": "exif_rgb_resize_short_side_bicubic_center_crop_v1 (PIL) on NSD 425x425 stimuli",
        "normalize": {"mean": processor.image_mean, "std": processor.image_std},
        "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "sample_stats": {"mean": float(stacked.mean()), "std": float(stacked.std())},
        "matches": "code/extra/siglip2 finetune/brainit-siglip2-unclip-train.py image_tokens()",
    }
    with open(out.replace(".pt", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    volume.commit()
    print(f"saved {len(targets)} targets -> {out}")
    print(json.dumps(meta["sample_stats"]))
    return meta


@app.function(gpu="H100", volumes={DATA_DIR: volume}, memory=32768, timeout=60 * 60)
def compare_preprocessing(subject: int = 1, n: int = 64):
    """Cosine between our cached targets (PIL bicubic) and tokens from modal_siglip2_eval.py's skimage resize."""
    import h5py
    import numpy as np
    import torch
    from scipy.io import loadmat
    from skimage.transform import resize
    from transformers import SiglipVisionModel

    encoder = SiglipVisionModel.from_pretrained(MODEL, revision=REVISION, cache_dir=f"{DATA_DIR}/external_models/huggingface",
                                                torch_dtype=torch.bfloat16).eval().cuda()
    targets = torch.load(f"{DATA_DIR}/siglip2_targets_cache/subj{subject:02d}_siglip2_targets.pt")
    shared = (loadmat(f"{DATA_DIR}/nsd_expdesign.mat")["subjectim"][0, :1000] - 1).astype(np.int64)[:n]
    stimuli = h5py.File(f"{DATA_DIR}/nsd_stimuli.hdf5", "r")["imgBrick"]
    skimage_px, pil_px = [], []
    for i in shared:
        pixels = resize(stimuli[int(i)], (SIZE, SIZE), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        skimage_px.append(torch.from_numpy(pixels).permute(2, 0, 1).float() / 255)
        pil_px.append(preprocess(stimuli[int(i)]))

    def encode(px):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            px = (torch.stack(px).cuda() - 0.5) / 0.5
            return torch.cat([encoder(pixel_values=px[k:k + BATCH].to(torch.bfloat16)).last_hidden_state.float().cpu()
                              for k in range(0, n, BATCH)])

    theirs, pil_again = encode(skimage_px), encode(pil_px)
    ours = torch.stack([targets[int(i)] for i in shared]).float()
    cos = torch.nn.functional.cosine_similarity
    token_cos = cos(ours, theirs, dim=-1)                       # [n, 256]
    flat_cos = cos(ours.flatten(1), theirs.flatten(1), dim=-1)
    pixel_diff = (torch.stack(skimage_px) - torch.stack(pil_px)).abs().mul(255)
    by_position = token_cos.mean(0)
    q = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75])
    out = {"n": n, "token_cos_mean": float(token_cos.mean()), "token_cos_min": float(token_cos.min()),
           "image_cos_mean": float(flat_cos.mean()), "image_cos_min": float(flat_cos.min()),
           "token_cos_quantiles_1_5_25_50_75": [round(float(v), 4) for v in token_cos.flatten().quantile(q)],
           "frac_tokens_cos_below_0.5": float((token_cos < 0.5).float().mean()),
           "frac_tokens_cos_below_0.9": float((token_cos < 0.9).float().mean()),
           "worst_positions_mean_cos": {int(p): round(float(by_position[p]), 4) for p in by_position.argsort()[:8]},
           "best_positions_mean_cos": round(float(by_position.sort().values[-8:].mean()), 4),
           "pil_reencode_vs_stored_image_cos_min": float(cos(ours.flatten(1), pil_again.flatten(1), dim=-1).min()),
           "pixel_absdiff_mean_0_255": float(pixel_diff.mean()), "pixel_absdiff_p99_0_255": float(pixel_diff.flatten().quantile(0.99)),
           "token_norm_ours_median_p99": [float(ours.norm(dim=-1).median()), float(ours.norm(dim=-1).flatten().quantile(0.99))]}
    print(out)
    return out


@app.local_entrypoint()
def main(subject: int = 1):
    print(extract.remote(subject))
