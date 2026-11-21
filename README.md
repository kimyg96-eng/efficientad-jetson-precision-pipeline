# EfficientAD-S precision policies and preprocessing concurrency on Jetson AGX Orin

Reproducibility materials for the study **Precision Policies and Preprocessing Concurrency for EfficientAD-S on Jetson AGX Orin**.

Repository: https://github.com/kimyg96-eng/efficientad-jetson-precision-pipeline

## Current status

Private repository with a draft v0.1.0 release. The owner-provided screenshots show acquisition, analysis, provenance and source_evidence folders and six attached evidence archives. These screenshots establish visible names and state, not independent verification of remote asset bytes. Local source copies were checked; download and verify remote assets before final release.

No public release or DOI has been issued. Researcher-original source files listed in MIT_LICENSED_FILES.csv are MIT-licensed; the copied upstream implementation retains Apache-2.0. See LICENSE for the exact boundary. External weights/images/engines are not redistributed here. Observed results concern EfficientAD-S, one Jetson AGX Orin, MODE_30W and four MVTec AD categories; they are not camera-system or hard-real-time guarantees.

## Contents

- `acquisition/step20a`: archived producer-count and pipeline-energy acquisition sources; 256-arm experiment.
- `acquisition/step21b`: archived synthetic fixed-arrival sources; 144 arms at 15/30/60 images/s.
- `acquisition/step22a`: producer/capacity experiment sources; final accepted data comprise 512 arms with recovery provenance.
- `acquisition/step23b`: archived dense-arrival sources; accepted fresh full session has 240 arms at 35/45/55/70/80 images/s. The pilot and interrupted 35 arms are excluded.
- `analysis/step23a`: exploratory deadline curves and finite-window queue accounting from existing traces.
- `analysis/step23c`: independently audited dense-grid summaries and on-time delivery curves.
- `analysis/step23d`: admission rejection categories and completed-image stage decomposition.
- `RAW_DATA_MANIFEST.csv`: expected raw-archive names, sizes and SHA-256 values. The six release-asset names and exact byte/hash values are recorded using their names in the draft, without local download suffixes.

## Raw evidence

The source archives are staged as private draft-release assets by the repository owner. This repository package includes no dataset images, model checkpoint bytes or TensorRT engine bytes. They remain private draft assets; no public availability is claimed.

The source supplement now includes training/export/validation/test wrappers and a source coverage index. Complete collected text evidence is in STEP24A_REVIEW.zip; STEP24B review adds independent lineage checks and previously prepared energy-audit outputs. See SOURCE_COVERAGE.md and THIRD_PARTY_NOTICES.md. Excluded dataset/weight/engine/map binaries and external reconstruction dependencies remain outside this code package. Earlier inference-workload raw traces are not asserted complete. Acquisition entry points use the original Jetson tree and hashes rather than a turnkey installer.

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

The owner has authorized MIT for the 88 researcher-original source files listed by path/hash. Third-party datasets, weights, libraries and copied code retain their respective terms. Documents and raw results are not implicitly relicensed by this source-code decision. Verify all source/raw asset coverage and external reconstruction dependencies; attach raw assets; create a versioned release and archive DOI; then update manuscript availability statements with real public links. Keep this repository private until that review is complete.

## Verify this source snapshot and the six evidence assets

The checksum list describes the intended research-material snapshot, not GitHub-generated ZIP wrappers or unrelated files such as .gitignore. Requires Python 3.11 or newer. Verification reads file bytes only and never imports or executes archived experiments.

```bash
python tools/verify_materials.py --root .
python tools/verify_materials.py --root . --assets /path/to/downloaded-release-assets
```

Save the six assets under the exact names in RAW_DATA_MANIFEST.csv. Renaming an archive does not change its SHA-256; do not rewrite or recompress original archives. Existing internal historical manifests remain unchanged.

## Source code license

Researcher-original Python and shell files are licensed under MIT only as enumerated in MIT_LICENSED_FILES.csv. The original source bytes are unchanged. The copied upstream anomalib module is excluded from that list, retains its Intel/Apache-2.0 header, and has the Apache-2.0 text in LICENSES/. Data, weights, engines, results and raw evidence archives are outside this grant. Licensing source does not create a public release or DOI and does not resolve external-weight conditions.
