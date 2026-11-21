# Source coverage and replay boundaries

The initial repository held pipeline acquisition and analysis. This supplement adds 47 source files from the existing training/export/build/validation/test lineage. Exact protocols, build logs, scalar metrics and original manifests are in the private release asset STEP24A_REVIEW.zip, not duplicated into Git. Failed attempts and retry source remain separately named; their presence is not scientific acceptance.

| Stage | Existing evidence family | Representation |
|---|---|---|
| Dependencies and external assets | STEP11A/B/C/D | Inventory, asset URLs/hashes, smoke source, anomalib 2.3.0 |
| Training | STEP11E and STEP14B | Bottle/Carpet and Capsule/Grid training source, summary and frozen path/hash roles |
| ONNX export | STEP11F and STEP14C | Export and output-parity source, version/protocol evidence |
| TensorRT policies | STEP11G/I/L/M/N, STEP13A, STEP14D, STEP16A, STEP19A | Build-source/protocol/log evidence, engine hashes; no engine bytes |
| Numerical validation | STEP11H/J/W, STEP13G, STEP14E, STEP16B, STEP19A | Scalar/map acceptance summaries and replay code; retries preserved |
| Independent test metrics | STEP11R, STEP12C, STEP13H, STEP14F, STEP15A/B/C | Test manifests, scoring/pixel/bootstrap source and derived scalar metrics; no image/map bytes |
| Pipeline raw traces | STEP20A/21B/22A/23B draft assets | Four previously verified complete experiment archives |
| Energy audit outputs | STEP16E/18B/19C | Previously prepared local analysis archives, separately sourced in STEP24B review; not collected anew on Jetson |

The new collection verified 890 collection hashes, 885 copied files, and 828 available original-manifest references with no mismatch. 71 entries were not in the collector's root-only comparison; nested manifests resolve some, including Capsule/Grid train.py. Unlisted files are not asserted to have original source-hash proof.

Training uses disjoint TRAIN and VAL image paths. Teacher channel statistics use TRAIN; VAL calibrates map quantiles after optimization and is reused for numerical screening and thresholds. It is not an independent calibration-audit set. Independent TEST paths remain distinct.

Scripts are archived exact sources and use the original evidence-tree layout, external packages, relative imports/AST-extracted helpers, assets and fixed engine/input hashes. They are not standalone installers. A complete numerical rerun needs externally acquired dataset/weights and rebuilt engines. Complete earlier inference-only raw traces and binary map caches are not claimed present in the four pipeline archives.
