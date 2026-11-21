import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

test_old, test_new, latency_old, latency_new, out = map(Path, sys.argv[1:])
categories = {
    "bottle": (83, 20), "carpet": (117, 28),
    "capsule": (132, 23), "grid": (78, 21),
}
engines = ("strict_fp32", "default_tf32", "automatic_fp16")

def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def write(name, header, rows):
    with (out / name).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)

def metrics(labels, values):
    return np.array([
        roc_auc_score(labels, values),
        average_precision_score(labels, values),
    ])

test_rows = read(test_old / "test_scores.csv") + read(test_new / "test_scores.csv")
latency_rows = (
    read(latency_old / "per_image_latency.csv")
    + read(latency_new / "per_image_latency.csv")
)
absolute, differences, stage_summary, budget_curve = [], [], [], []
budgets = sorted({16.67, 33.33, 66.67, *range(20, 181)})
stages = (
    "read_ms", "decode_preprocess_ms",
    "transfer_infer_score_decision_ms", "png_to_decision_ms",
)
repeats, seed = 5000, 20261118

for category, (expected_n, expected_normal) in categories.items():
    rows = [row for row in test_rows if row["category"] == category]
    paths = {row["path"] for row in rows}
    if len(rows) != expected_n or len(paths) != expected_n:
        raise ValueError(f"Unexpected test inventory: {category}")
    if any(row["label"] not in {"0", "1"} for row in rows):
        raise ValueError(f"Invalid test label: {category}")
    labels = np.array([int(row["label"]) for row in rows])
    normal, anomalous = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    if len(normal) != expected_normal:
        raise ValueError(f"Unexpected normal count: {category}")

    strict_column = "original_no_tf32" if category in {"bottle", "carpet"} else "strict_fp32"
    scores = np.asarray([
        [float(row[column]) for row in rows]
        for column in (strict_column, "default_tf32", "automatic_fp16")
    ])
    if not np.isfinite(scores).all():
        raise ValueError(f"Nonfinite test scores: {category}")
    point = np.array([metrics(labels, values) for values in scores])
    for engine, values in zip(engines, point, strict=True):
        absolute.append([category, engine, expected_n, *values])

    rng = np.random.default_rng(seed)
    samples = np.empty((repeats, 3, 2))
    for i in range(repeats):
        idx = np.concatenate([
            rng.choice(normal, size=len(normal), replace=True),
            rng.choice(anomalous, size=len(anomalous), replace=True),
        ])
        samples[i] = [metrics(labels[idx], values[idx]) for values in scores]

    for a, b in ((0, 1), (1, 2)):
        delta = 100 * (point[b] - point[a])
        lo, hi = np.percentile(100 * (samples[:, b] - samples[:, a]), [2.5, 97.5], axis=0)
        score_delta = np.abs(scores[b] - scores[a])
        i = int(np.argmax(score_delta))
        ref_span = float(np.ptp(scores[a]))
        relative = 100 * score_delta[i] / ref_span if ref_span else ""
        differences.append([
            category, engines[a], engines[b],
            delta[0], lo[0], hi[0], delta[1], lo[1], hi[1],
            score_delta[i], scores[a, i], ref_span, relative,
        ])

    for engine in engines[1:]:
        rounds = []
        for run in range(1, 7):
            group = [
                row for row in latency_rows
                if row["category"] == category and row["engine"] == engine
                and int(row["round"]) == run
            ]
            if len(group) != expected_n or {row["path"] for row in group} != paths:
                raise ValueError(f"Incomplete timing inventory: {category}/{engine}/{run}")
            times = np.array([[float(row[key]) for key in stages] for row in group])
            if (not np.isfinite(times).all() or (times < 0).any()
                    or not np.allclose(times[:, :3].sum(axis=1), times[:, 3], rtol=0, atol=1e-6)):
                raise ValueError(f"Invalid timing stages: {category}/{engine}/{run}")
            rounds.append(times)

        values = np.stack(rounds)
        for j, stage in enumerate(stages):
            p50, p95 = np.percentile(values[:, :, j], [50, 95], axis=1)
            stage_summary.append([category, engine, stage, expected_n * 6,
                                  p50.mean(), p95.mean()])
        for budget in budgets:
            rate = 100 * (values[:, :, 3] > budget).mean(axis=1)
            budget_curve.append([category, engine, budget, rate.mean(), rate.min(), rate.max()])

write("image_metrics.csv", ["category", "engine", "n", "auroc", "ap"], absolute)
write("image_paired_differences.csv", [
    "category", "reference", "candidate",
    "auroc_delta_pp", "auroc_ci_low_pp", "auroc_ci_high_pp",
    "ap_delta_pp", "ap_ci_low_pp", "ap_ci_high_pp",
    "max_abs_score_delta", "reference_score_at_max_delta",
    "reference_test_score_span", "max_abs_delta_pct_of_reference_span",
], differences)
write("latency_stages.csv", [
    "category", "engine", "stage", "timed_calls", "mean_round_p50_ms", "mean_round_p95_ms",
], stage_summary)
write("latency_budget_curve.csv", [
    "category", "engine", "budget_ms", "mean_exceedance_pct",
    "min_round_exceedance_pct", "max_round_exceedance_pct",
], budget_curve)
print(f"IMAGE_METRICS={len(absolute)} / PAIRED_COMPARISONS={len(differences)} "
      f"/ BOOTSTRAP={repeats} / SEED={seed}")
print(f"LATENCY_STAGE_ROWS={len(stage_summary)} / BUDGET_CURVE_ROWS={len(budget_curve)}")
