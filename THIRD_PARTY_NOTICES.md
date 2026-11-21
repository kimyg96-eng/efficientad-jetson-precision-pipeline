# Third party provenance and licensing scope

This document records third-party provenance. Researcher-original code is separately licensed by LICENSE and MIT_LICENSED_FILES.csv; no third-party rights are expanded by that grant.

## Implementation source

The collected implementation is anomalib 2.3.0. Its package metadata lists Apache License 2.0, and the copied upstream EfficientAD module preserves Intel copyright and SPDX-License-Identifier: Apache-2.0. The exact upstream file appears in source_evidence/STEP11B_EFFICIENTAD_ASSETS_Cn7KT6Qn/upstream_efficientad_lightning_model.py and in the complete source archive. Preserve those notices; the Apache-2.0 license text is included in LICENSES/Apache-2.0.txt, and any additional applicable upstream notices must remain intact. Researcher-original experiment wrappers and analysis are distinct from that upstream implementation.

Primary package record: https://pypi.org/project/anomalib/2.3.0/
Apache license text: https://www.apache.org/licenses/LICENSE-2.0.txt

## Teacher weights

Frozen weight archive URL and SHA-256 come from STEP11B/summary.json. Dataset/checkpoint/engine bytes are not included. The upstream licensing question https://github.com/open-edge-platform/anomalib/issues/3710 was still open when checked on 21 November 2026; the observed maintainer response says clarification is pending. This is an unresolved provenance/licensing question, not a finding that academic use/publication is prohibited. Do not assume implementation-source Apache-2.0 automatically licenses all teacher weights. No weight redistribution permission is claimed.

Archive SHA-256: c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a
Used teacher-member SHA-256: a16ded54719674435576aee641152616a640dfc6dc2b83115dab6e226610ae7d
Weight archive URL: https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip

## Images and auxiliary data

MVTec AD's official page states CC BY-NC-SA 4.0. No images/masks are redistributed here. Original dataset conditions are not replaced by a repository source-code license.
https://www.mvtec.com/research-teaching/datasets/mvtec-ad

The auxiliary ImageNette archive URL/hash is preserved from the frozen upstream asset declaration. Its ImageNet-derived image terms require separate review; no new image redistribution license is inferred from package source licensing. No ImageNette archive bytes are included.
URL: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
SHA-256: 6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd

PyTorch, torchvision, Pillow, numpy, ONNX, ONNX Runtime and NVIDIA software retain their own terms. This preparation does not redistribute their installed packages or claim one repository license covers all dependencies.
