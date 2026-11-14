# CortexDecoder: fMRI-to-Image Reconstruction from Cortical Flatmaps

Reconstructs viewed images from fMRI using cortical flatmaps. A frozen CortexMAE encoder turns visual-cortex flatmap patches into brain tokens, two cross-attention bridges predict VGG (low-level) and OpenCLIP bigG (semantic) features, and a frozen SDXL-unCLIP model generates the image from a Deep Image Prior init. Only the bridges are trained (83M parameters, about 3 H100-hours).

**Paper:** [bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.10.09.757766v1)

![CortexDecoder reconstructions](papers/IEEE_paper/images/title5.jpg)

## Results

NSD Subject 1, 998 shared test images, one sample per image. NeuroPictor is the other cortical-flatmap method; its frozen-encoder ablation matches our setting (Subject 1 only, frozen fMRI encoder). Bold marks the better of the matched pair; the full NeuroPictor is shown for reference.

| Method | PixCorr↑ | SSIM↑ | Alex(2)↑ | Alex(5)↑ | Incep↑ | CLIP↑ | Eff↓ | SwAV↓ |
|---|---|---|---|---|---|---|---|---|
| NeuroPictor, frozen encoder | 0.220 | 0.342 | 97.5% | 98.8% | 93.8% | 92.8% | 0.657 | 0.360 |
| **CortexDecoder** | **0.366** | **0.413** | **98.1%** | **99.1%** | **94.9%** | **93.5%** | **0.628** | 0.360 |
| *NeuroPictor, multi-subject + fine-tuned encoder* | *0.277* | *0.385* | *98.8%* | *99.3%* | *96.2%* | *94.5%* | *0.619* | *0.334* |

The full comparison with voxel-based methods is in Table 1 of the paper, and per-run metrics are in [`results/v2/`](results/v2/).

## Usage

Runs on [Modal](https://modal.com) with the `braid2-flatmap-data` volume. Every call is detached, so runs survive the laptop closing. The commands below reproduce the paper's final model; any field in [`config.py`](code/v2/cortexdecoder/core/config.py) can be passed as an override.

```bash
modal deploy code/v2/modal_app.py
SPAWN="python3 code/v2/scripts/spawn_runs.py"
ROI='"brain_tokens": "flatmap_tokens_roi-nsdgeneral_fp16"'   # visual-cortex tokens from scripts/generate_roi_tokens.py

$SPAWN vgg  "{\"timestamp\": \"<run-id>\", $ROI}"   # low-level bridge + DIP images
$SPAWN clip "{\"timestamp\": \"<run-id>\", $ROI}"   # semantic bridge, MSE
$SPAWN clip "{\"timestamp\": \"<run-id>\", $ROI, \"decoder_loss_weight\": 16.0, \"lr\": 5e-5, \"epochs\": 5, \"warmup_epochs\": 0, \"init_checkpoint\": \"<clip-run-dir>/clip_best.pt\"}"   # decoder-aware fine-tune
$SPAWN final "{\"timestamp\": \"<run-id>\", $ROI, \"vgg_brain_tokens\": \"flatmap_tokens_roi-nsdgeneral_fp16\", \"decoder_loss_weight\": 16.0}"   # SDXL-unCLIP + eval
```

`<clip-run-dir>` is the output directory printed by the MSE run.

## Layout

```
code/v2/   pipeline (Python package + Modal app) and figure scripts
code/v1/   original notebooks
papers/    IEEE camera-ready (IEEE_paper/), AAAI workshop version (AAAI_workshop_paper/)
results/   evaluation tables (v1: whole cortex, best of 10; v2: current runs)
```

## Citation

```bibtex
@article{imototar2026cortexdecoder,
  title   = {CortexDecoder: fMRI-to-image Reconstruction from Cortical Flatmaps},
  author  = {Imoto-Tar, Teo and Fung, Maxwell and Gurha, Arnav},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.10.09.757766}
}
```
