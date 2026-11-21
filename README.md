# EfficientAD-S precision policies and preprocessing concurrency on Jetson AGX Orin

Reproducibility materials for the study **Precision Policies and Preprocessing Concurrency for EfficientAD-S on Jetson AGX Orin**.

Repository: https://github.com/kimyg96-eng/efficientad-jetson-precision-pipeline

## Current status

Prepared for a private repository. This package has not been pushed by the assistant. No public release, DOI or source-code license has been issued. Observed results concern EfficientAD-S, one Jetson AGX Orin, MODE_30W and four MVTec AD categories. They are not camera-system or hard-real-time guarantees.

## Contents

- `acquisition/step20a`: archived producer-count and pipeline-energy acquisition sources; 256-arm experiment.
- `acquisition/step21b`: archived synthetic fixed-arrival sources; 144 arms at 15/30/60 images/s.
- `acquisition/step22a`: producer/capacity experiment sources; final accepted data comprise 512 arms with recovery provenance.
- `acquisition/step23b`: archived dense-arrival sources; accepted fresh full session has 240 arms at 35/45/55/70/80 images/s. The pilot and interrupted 35 arms are excluded.
- `analysis/step23a`: exploratory deadline curves and finite-window queue accounting from existing traces.
- `analysis/step23c`: independently audited dense-grid summaries and on-time delivery curves.
- `analysis/step23d`: admission rejection categories and completed-image stage decomposition.
- `RAW_DATA_MANIFEST.csv`: expected raw-archive names, sizes and SHA-256 values. Large archives are deliberately separate from this code upload.

## Raw evidence

The source archives remain separately held by the authors. This repository package includes no dataset images, model checkpoint bytes or TensorRT engine bytes. Raw trace archives should be attached to a versioned release or an archive service after redistribution review. Do not claim public availability before these assets and links exist.

The original training, ONNX export, engine-build and complete numerical-evaluation evidence is not included in this initial upload. A complete paper-wide release must add or link those sources. Existing acquisition entry points reference the original Jetson evidence tree and frozen hashes; they are archived exact source, not portable turnkey installers. Do not run clock-control acquisition scripts on another system without reviewing the expected hardware paths and recovery handling.

## Reproduce derived results

Extract the named raw ZIPs into separate directories, preserving their internal paths. Python dependencies for analysis are numpy and matplotlib. Acquisition environments have additional pinned CUDA, TensorRT, PyTorch, torchvision and Pillow requirements recorded in the original evidence; no generic version substitution is claimed equivalent.

```bash
python analysis/step23c/analyze.py /path/to/extracted_STEP23B_FULL /path/to/output23c
python analysis/step23d/analyze.py /path/to/extracted_STEP23B_FULL --out /path/to/output23d
python analysis/step23a/deadline_analysis.py /path/to/extracted_STEP21B_FULL --capacity-source /path/to/extracted_STEP22A_COMPLETE --out /path/to/output23a
```

The analysis scripts intentionally validate fixed accepted corpus dimensions. The STEP23C numerical checks use the included power integration implementation and retain its 150 ms maximum integration-gap rule. SHA256SUMS.txt describes this code package; each raw archive has its original internal hash manifest.

## Measurement interpretation

- Delivery fraction includes every offered arrival; dropped inputs never become on-time decisions.
- Arrival rate and decision deadline are independent. The original deadline sweep is post hoc.
- Runs are repeatability units; repeated image calls are not independent accuracy cases. Ranges are observed run minima/maxima.
- New and earlier arrival sessions are not pooled. Zero-drop means all six rounds at a tested rate, not a guaranteed rate limit.
- Pipeline energy is three onboard rails, not wall-socket energy. Inference-only and pipeline clock/boundary conditions differ.
- Host consumer service fraction is not hardware GPU utilization. Capacity rejection is an admission category, not a unique causal bottleneck diagnosis.
- Little's law addresses means; finite-window accounting does not predict p95.

## Licensing and public-release checklist

No license is granted by this preparation step. Third-party datasets, weights, libraries and copied code retain their respective terms. Review source provenance and select an appropriate license before public release. Add/link the missing early evidence; attach raw assets; create a versioned release and archive DOI; then update manuscript availability statements with real public links. Keep this repository private until that review is complete.
