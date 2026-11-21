import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

audit, data, out = (Path(arg).resolve() for arg in sys.argv[1:])
stages = {
    "e": "STEP11E_EFFICIENTAD_TRAIN_ZeV4ionZ",
    "g": "STEP11G_EFFICIENTAD_TRT_FP32_vS97CAlx",
    "i": "STEP11I_EFFICIENTAD_TRT_FP16_4xaX3g44",
    "l": "STEP11L_EFFICIENTAD_CONV_FP16_2aQndiJ2",
    "m": "STEP11M_EFFICIENTAD_BRANCH_FP16_oetzGryV",
    "n": "STEP11N_EFFICIENTAD_AE_SPLIT_FP16_B1HhoFFr",
    "r": "STEP11R_EFFICIENTAD_TEST_nqm19zQT",
    "u": "STEP11U_EFFICIENTAD_DECISION_ANALYSIS_bR5Nqxwn",
    "v": "STEP11V_EFFICIENTAD_SCREENING_TABLE_lP0iBfrv",
}
stages = {key: audit / name for key, name in stages.items()}
for directory in stages.values():
    subprocess.run(
        ["sha256sum", "-c", "--quiet", "SHA256SUMS.txt"],
        cwd=directory, check=True,
    )
    (out / f"{directory.name.split('_')[0]}_SOURCE_SHA256SUMS.txt").write_bytes(
        (directory / "SHA256SUMS.txt").read_bytes()
    )

# Load the verified STEP11R helpers with the same TensorRT and preprocessing setup.
trt_site = subprocess.check_output([
    "/usr/bin/python3", "-c",
    "import pathlib,tensorrt; print(pathlib.Path(tensorrt.__file__).resolve().parent.parent)",
], text=True).strip()
source = stages["r"] / "evaluate.py"
marker = "\nresults = {}\n"
code = source.read_text()
if code.count(marker) != 1:
    raise ValueError("Unexpected STEP11R evaluation script")
(out / "test_manifest.csv").write_bytes((stages["r"] / "test_manifest.csv").read_bytes())
saved_argv = sys.argv
sys.argv = [str(source), str(stages["g"]), str(stages["n"]), str(data), str(out), trt_site]
scope = {"__name__": "step11r_helpers"}
try:
    exec(compile(code.split(marker, 1)[0], str(source), "exec"), scope)
finally:
    sys.argv = saved_argv

torch = scope["torch"]
Image = scope["Image"]
io = scope["io"]
open_engine = scope["open_engine"]
transform = scope["transform"]
checked_bytes = scope["checked_bytes"]

def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

protocol = json.loads((stages["u"] / "protocol.json").read_text())
if (protocol["operating_points"] != {"primary_max": 1.0, "sensitivity_p95": 0.95}
        or protocol["quantile_method"] != "linear"
        or protocol["decision_rule"] != "score > threshold"):
    raise ValueError("Unexpected STEP11U decision policy")

multipliers = (0.5, 1.0, 2.0, 5.0)
(out / "protocol.json").write_text(json.dumps({
    "analysis": "Additional analysis on the existing test set",
    "decision_policy_source": str(stages["u"] / "protocol.json"),
    "decision_rule": protocol["decision_rule"],
    "operating_points": protocol["operating_points"],
    "quantile_method": protocol["quantile_method"],
    "threshold_policies": ["own", "shared_candidate"],
    "base_tolerance": "1e-4 + 1e-3 * abs(TensorRT FP32 output)",
    "tolerance_multipliers": multipliers,
    "historical_screening_multiplier": 1.0,
}, indent=2) + "\n")
subprocess.run(["sha256sum", "protocol.json"], cwd=out, check=True)

train = read_rows(stages["e"] / "input_sha256.csv")
test = read_rows(out / "test_manifest.csv")
original = {
    (row["category"], row["configuration"]): row
    for row in read_rows(stages["v"] / "screening_table.csv")
}
configs = {
    "automatic_fp16_enabled": (stages["i"], "fp16"),
    "all_conv_fp16": (stages["l"], "conv_fp16"),
    "teacher_conv_fp16": (stages["m"] / "teacher", "conv_fp16"),
    "student_conv_fp16": (stages["m"] / "student", "conv_fp16"),
    "autoencoder_conv_fp16": (stages["m"] / "autoencoder", "conv_fp16"),
    "ae_encoder_conv_fp16": (stages["n"] / "encoder", "conv_fp16"),
    "ae_decoder_conv_fp16": (stages["n"] / "decoder", "conv_fp16"),
}

def load_image(path, digest, subtree):
    payload = checked_bytes(path, digest, subtree)
    with Image.open(io.BytesIO(payload)) as image:
        return transform(image.convert("RGB")).unsqueeze(0).contiguous()

def run_engine(path, category, val_images, test_images):
    engine, context, buffers = open_engine(path, category)
    stream = torch.cuda.Stream()
    val_scores, val_maps, test_scores = [], [], []
    for images, scores, save_maps in (
        (val_images, val_scores, True), (test_images, test_scores, False)
    ):
        for image in images:
            with torch.cuda.stream(stream):
                buffers["image"].copy_(image)
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError(f"Inference failed: {path}")
            stream.synchronize()
            scores.append(buffers["pred_score"].item())
            if save_maps:
                val_maps.append(buffers["anomaly_map"].cpu().numpy()[0, 0])
    outputs = (np.asarray(val_scores), np.stack(val_maps), np.asarray(test_scores))
    if any(not np.isfinite(values).all() for values in outputs):
        raise ValueError(f"Non-finite inference output: {path}")
    return outputs

def confusion(labels, predictions):
    return {
        "tp": int(np.count_nonzero(labels & predictions)),
        "tn": int(np.count_nonzero(~labels & ~predictions)),
        "fp": int(np.count_nonzero(~labels & predictions)),
        "fn": int(np.count_nonzero(labels & ~predictions)),
    }

screening, decisions, image_scores, image_violations = [], [], [], []
for category, counts in {"bottle": (21, 20, 63), "carpet": (28, 28, 89)}.items():
    val_rows = [r for r in train if r["category"] == category and r["role"] == "VAL"]
    test_rows = [r for r in test if r["category"] == category]
    labels_int = [int(r["anomalous"]) for r in test_rows]
    if (len(val_rows) != counts[0] or len(test_rows) != sum(counts[1:])
            or (labels_int.count(0), labels_int.count(1)) != counts[1:]
            or len({r["path"] for r in val_rows}) != counts[0]
            or len({r["image_path"] for r in test_rows}) != len(test_rows)):
        raise ValueError(f"Unexpected inventory: {category}")
    labels = np.asarray(labels_int, dtype=bool)
    val_images = [
        load_image(r["path"], r["sha256"], f"{category}/train/good")
        for r in val_rows
    ]
    test_images = [
        load_image(r["image_path"], r["image_sha256"], f"{category}/test")
        for r in test_rows
    ]

    stem = f"{category}_efficientad_s"
    ref_val, ref_maps, ref_test = run_engine(
        stages["g"] / f"{stem}_fp32.engine", category, val_images, test_images,
    )
    ref_maps = ref_maps.astype(np.float64)
    score_tol = 1e-4 + 1e-3 * np.abs(ref_val)
    map_tol = 1e-4 + 1e-3 * np.abs(ref_maps)

    for name, (directory, suffix) in configs.items():
        cand_val, cand_maps, cand_test = run_engine(
            directory / f"{stem}_{suffix}.engine", category, val_images, test_images,
        )
        score_error = np.abs(cand_val - ref_val)
        map_error = np.abs(cand_maps.astype(np.float64) - ref_maps)
        historical = original[category, name]
        for factor in multipliers:
            score_bad = score_error > factor * score_tol
            pixel_bad = np.count_nonzero(map_error > factor * map_tol, axis=(1, 2))
            current = {
                "scores_outside_tolerance": int(score_bad.sum()),
                "pixels_outside_tolerance": int(pixel_bad.sum()),
                "images_outside_tolerance": int(np.count_nonzero(score_bad | (pixel_bad > 0))),
            }
            screening.append({
                "category": category, "configuration": name, "multiplier": factor,
                "val_images": len(val_rows), "map_pixels": int(ref_maps.size), **current,
                "status": "FAIL" if current["images_outside_tolerance"] else "PASS",
                "matches_historical_counts": (
                    all(value == int(historical[key]) for key, value in current.items())
                    if factor == 1.0 else ""
                ),
            })
            image_violations.extend({
                "category": category, "configuration": name, "multiplier": factor,
                "image_path": row["path"], "score_outside_tolerance": int(score_bad[i]),
                "pixels_outside_tolerance": int(pixel_bad[i]),
            } for i, row in enumerate(val_rows))

        image_scores.extend({
            "category": category, "configuration": name, "image_path": row["image_path"],
            "anomalous": labels_int[i], "fp32_score": float(ref_test[i]),
            "candidate_score": float(cand_test[i]),
        } for i, row in enumerate(test_rows))

        for point, q in protocol["operating_points"].items():
            t_ref, t_cand = (
                float(np.quantile(values, q, method="linear"))
                for values in (ref_val, cand_val)
            )
            for policy in ("own", "shared_candidate"):
                ref_threshold = t_ref if policy == "own" else t_cand
                ref_pred = ref_test > ref_threshold
                cand_pred = cand_test > t_cand
                decisions.append({
                    "category": category, "configuration": name,
                    "historical_screening": historical["status"],
                    "operating_point": point, "policy": policy,
                    "fp32_threshold": ref_threshold, "candidate_threshold": t_cand,
                    "test_images": len(test_rows),
                    "disagreements": int(np.count_nonzero(ref_pred != cand_pred)),
                    "new_false_positives": int(np.count_nonzero(~labels & ~ref_pred & cand_pred)),
                    "new_false_negatives": int(np.count_nonzero(labels & ref_pred & ~cand_pred)),
                    **{f"fp32_{k}": v for k, v in confusion(labels, ref_pred).items()},
                    **{f"candidate_{k}": v for k, v in confusion(labels, cand_pred).items()},
                })
        print(f"DONE {category}/{name}", flush=True)

write_rows(out / "tolerance_sensitivity.csv", screening)
write_rows(out / "val_image_violations.csv", image_violations)
write_rows(out / "test_decisions.csv", decisions)
write_rows(out / "test_image_scores.csv", image_scores)

summary = {
    "pass_counts_by_multiplier": {
        category: {
            str(factor): sum(
                r["category"] == category and r["multiplier"] == factor and r["status"] == "PASS"
                for r in screening
            )
            for factor in multipliers
        }
        for category in ("bottle", "carpet")
    },
    "historical_count_mismatches": [
        f"{r['category']}/{r['configuration']}"
        for r in screening
        if r["multiplier"] == 1.0 and not r["matches_historical_counts"]
    ],
    "primary_decisions": [
        {key: r[key] for key in (
            "category", "configuration", "historical_screening", "policy",
            "disagreements", "new_false_positives", "new_false_negatives",
        )}
        for r in decisions if r["operating_point"] == "primary_max"
    ],
}
text = json.dumps(summary, indent=2, allow_nan=False)
(out / "summary.json").write_text(text + "\n")
print(text)
