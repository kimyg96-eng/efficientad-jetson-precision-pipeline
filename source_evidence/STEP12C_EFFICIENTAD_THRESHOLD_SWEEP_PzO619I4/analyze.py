import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

audit, out = map(Path, sys.argv[1:])
names = {
    "j": "STEP11J_EFFICIENTAD_TRT_FP16_VAL_kygZcMS9",
    "l": "STEP11L_EFFICIENTAD_CONV_FP16_2aQndiJ2",
    "m": "STEP11M_EFFICIENTAD_BRANCH_FP16_oetzGryV",
    "n": "STEP11N_EFFICIENTAD_AE_SPLIT_FP16_B1HhoFFr",
    "w": "STEP11W_EFFICIENTAD_SCREENING_SENSITIVITY_rP2PQKD4",
}
dirs = {key: audit / name for key, name in names.items()}
for directory in dirs.values():
    subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS.txt"],
                   cwd=directory, check=True)
    step = directory.name.split("_", 1)[0]
    (out / f"{step}_SOURCE_SHA256SUMS.txt").write_bytes(
        (directory / "SHA256SUMS.txt").read_bytes()
    )

configs = {
    "automatic_fp16_enabled": (dirs["j"], "trt_fp16"),
    "all_conv_fp16": (dirs["l"], "conv_fp16"),
    "teacher_conv_fp16": (dirs["m"] / "teacher", "conv_fp16"),
    "student_conv_fp16": (dirs["m"] / "student", "conv_fp16"),
    "autoencoder_conv_fp16": (dirs["m"] / "autoencoder", "conv_fp16"),
    "ae_encoder_conv_fp16": (dirs["n"] / "encoder", "conv_fp16"),
    "ae_decoder_conv_fp16": (dirs["n"] / "decoder", "conv_fp16"),
}
points = {"primary_max": 1.0, "sensitivity_p95": 0.95}
seed, repeats = 20261117, 2000
(out / "protocol.json").write_text(json.dumps({
    "analysis": "Retrospective analysis of existing EfficientAD-S scores",
    "shared_sweep": "All observed test score breakpoints plus a threshold below the minimum",
    "decision_rule": "score > threshold",
    "operating_points": points,
    "quantile_method": "linear",
    "bootstrap": "Paired normal VAL resampling with test images held fixed",
    "seed": seed,
    "replicates": repeats,
}, indent=2) + "\n")

def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def column(rows, keys, source):
    present = [key for key in keys if key in rows[0]]
    if len(present) != 1:
        raise ValueError(f"{source}: expected one of {keys}, found {list(rows[0])}")
    values = np.array([float(row[present[0]]) for row in rows])
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: non-finite scores")
    return values

def distribution(values):
    q = np.quantile(values, [0, 0.25, 0.5, 0.75, 1], method="linear")
    return dict(zip(("min", "q25", "median", "q75", "max"), q.tolist()))

test_rows = read(dirs["w"] / "test_image_scores.csv")
previous = read(dirs["w"] / "test_decisions.csv")
prior_by_key = {
    (r["category"], r["configuration"], r["operating_point"], r["policy"]): r
    for r in previous
}
rng = np.random.default_rng(seed)
sweep, margins, distributions, bootstrap, summary = [], [], [], [], []

for category, (nval, nnormal, nanomaly) in {
    "bottle": (21, 20, 63), "carpet": (28, 28, 89)
}.items():
    indices = rng.integers(0, nval, size=(repeats, nval))
    for config, (directory, suffix) in configs.items():
        source = directory / f"{category}_{suffix}_val.csv"
        val = read(source)
        test = [r for r in test_rows if r["category"] == category
                and r["configuration"] == config]
        if (len(val) != nval
                or len({r["path"] for r in val}) != nval
                or any(not r["path"].startswith(f"{category}/train/good/") for r in val)
                or len(test) != nnormal + nanomaly
                or len({r["image_path"] for r in test}) != len(test)):
            raise ValueError(f"{category}/{config}: unexpected inventory")

        rv = column(val, ("fp32_score",), source)
        cv = column(val, ("fp16_score", "conv_fp16_score"), source)
        rt = column(test, ("fp32_score",), config)
        ct = column(test, ("candidate_score",), config)
        labels = np.array([int(r["anomalous"]) for r in test])
        if ((labels == 0).sum(), (labels == 1).sum()) != (nnormal, nanomaly):
            raise ValueError(f"{category}/{config}: unexpected label counts")
        normal, abnormal = labels == 0, labels == 1
        key = {"category": category, "configuration": config}

        for engine, validation, scores in (("fp32", rv, rt), ("candidate", cv, ct)):
            for partition, values in (
                ("normal_val", validation), ("normal_test", scores[normal]),
                ("anomalous_test", scores[abnormal]),
            ):
                distributions.append({
                    **key, "engine": engine, "partition": partition,
                    "n": int(values.size), **distribution(values),
                    "normal_test_above_val_max": (
                        int((values > validation.max()).sum())
                        if partition == "normal_test" else ""
                    ),
                })

        cuts = np.unique(np.r_[rt, ct])
        cuts = np.r_[np.nextafter(cuts[0], -np.inf), cuts]
        changed = (rt[None, :] > cuts[:, None]) != (ct[None, :] > cuts[:, None])
        total = changed.sum(axis=1)
        for i, threshold in enumerate(cuts):
            sweep.append({
                **key, "threshold": float(threshold),
                "next_breakpoint": float(cuts[i + 1]) if i + 1 < len(cuts) else "",
                "disagreements": int(total[i]),
                "normal_disagreements": int(changed[i, normal].sum()),
                "anomalous_disagreements": int(changed[i, abnormal].sum()),
            })

        tau = float(kendalltau(rt, ct, variant="b").statistic)
        summary.append({
            **key, "max_abs_score_delta": float(np.max(np.abs(ct - rt))),
            "kendall_tau_b": tau if np.isfinite(tau) else None,
            "peak_shared_threshold_disagreements": int(total.max()),
            "test_images": len(test),
            "fp32_normal_test_above_own_val_max": int((rt[normal] > rv.max()).sum()),
            "candidate_normal_test_above_own_val_max": int((ct[normal] > cv.max()).sum()),
        })

        for point, q in points.items():
            ref_thr, cand_thr = (
                float(np.quantile(values, q, method="linear"))
                for values in (rv, cv)
            )
            ref_boot, cand_boot = (
                np.quantile(values[indices], q, axis=1, method="linear")
                for values in (rv, cv)
            )
            for policy in ("own", "shared_candidate"):
                ref_cut = ref_thr if policy == "own" else cand_thr
                rmargin, cmargin = rt - ref_cut, ct - cand_thr
                disagreement = int(np.count_nonzero((rmargin > 0) != (cmargin > 0)))
                saved = prior_by_key[category, config, point, policy]
                if (disagreement != int(saved["disagreements"])
                        or ref_cut != float(saved["fp32_threshold"])
                        or cand_thr != float(saved["candidate_threshold"])):
                    raise ValueError(f"STEP11W mismatch: {category}/{config}/{point}/{policy}")

                nearest = np.argsort(
                    np.minimum(np.abs(rmargin), np.abs(cmargin)), kind="stable"
                )[:5]
                for rank, i in enumerate(nearest, 1):
                    margins.append({
                        **key, "operating_point": point, "policy": policy, "rank": rank,
                        "image_path": test[i]["image_path"],
                        "anomalous": int(labels[i]), "score_delta": float(ct[i] - rt[i]),
                        "fp32_margin": float(rmargin[i]), "candidate_margin": float(cmargin[i]),
                    })

                ref_cuts = ref_boot if policy == "own" else cand_boot
                boot_counts = np.count_nonzero(
                    (rt[None, :] > ref_cuts[:, None])
                    != (ct[None, :] > cand_boot[:, None]), axis=1,
                )
                bootstrap.append({
                    **key, "operating_point": point, "policy": policy,
                    "observed_disagreements": disagreement,
                    "replicates_with_disagreement": int(np.count_nonzero(boot_counts)),
                    "replicates": repeats,
                    "median_disagreements": float(np.median(boot_counts)),
                    "max_disagreements": int(boot_counts.max()),
                })

write(out / "threshold_sweep.csv", sweep)
write(out / "nearest_threshold_margins.csv", margins)
write(out / "score_distributions.csv", distributions)
write(out / "val_threshold_bootstrap.csv", bootstrap)
write(out / "summary.csv", summary)

for row in summary:
    print(f"{row['category']}/{row['configuration']} "
          f"peak_shared_disagreements={row['peak_shared_threshold_disagreements']}/{row['test_images']} "
          f"max_abs_delta={row['max_abs_score_delta']:.9g} tau_b={row['kendall_tau_b']}")
print("STEP11W_OPERATING_POINTS_REPRODUCED=56/56")
