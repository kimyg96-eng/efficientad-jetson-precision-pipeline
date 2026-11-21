import csv
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

src, data, out = (Path(arg).resolve(strict=True) for arg in sys.argv[1:])
engines = ("strict_fp32", "default_tf32", "automatic_fp16")
categories = {"bottle": (20, 63), "carpet": (28, 89),
              "capsule": (23, 109), "grid": (21, 57)}
contrasts = ((0, 1), (1, 2))
repeats, seed = 5000, 20261118
fine_bins, coarse_bins = 65536, 16384

def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def checked_file(relative, digest, subtree):
    path = (data / relative).resolve(strict=True)
    if not path.is_relative_to(data / subtree):
        raise ValueError(f"Unexpected path: {relative}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"SHA-256 mismatch: {relative}")
    return payload

def grouped_metrics(negative, positive):
    # The bins are ordered from highest to lowest score.
    tp = np.cumsum(positive, axis=1)
    fp = np.cumsum(negative, axis=1)
    p, n = tp[:, -1], fp[:, -1]
    if np.any(p <= 0) or np.any(n <= 0):
        raise ValueError("A resample lacks positive or negative pixels")
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    ap = (positive * precision).sum(axis=1) / p
    auc = (positive * (n[:, None] - fp + 0.5 * negative)).sum(axis=1) / (p * n)
    return np.column_stack((auc, ap))

def evaluate(hist, weights):
    result = np.empty((len(weights), 2))
    for start in range(0, len(weights), 32):
        stop = min(start + 32, len(weights))
        w = weights[start:stop]
        result[start:stop] = grouped_metrics(
            (w @ hist[0])[:, ::-1], (w @ hist[1])[:, ::-1],
        )
    return result

manifest = read(src / "test_manifest.csv")
points = {
    (row["category"], row["engine"]):
        np.array([float(row["pixel_auroc"]), float(row["pixel_ap"])])
    for row in read(src / "pixel_metrics.csv")
}
output = []

for category, expected in categories.items():
    rows = [row for row in manifest if row["category"] == category]
    labels = np.asarray([int(row["anomalous"]) for row in rows])
    normal, anomalous = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    if (len(normal), len(anomalous)) != expected or len(rows) != sum(expected):
        raise ValueError(f"Unexpected test inventory: {category}")
    paths = [row["image_path"] for row in rows]
    maps = []
    for engine in engines:
        prefix = src / f"{category}_{engine}"
        index = read(prefix.with_suffix(".scores.csv"))
        if ([row["image_path"] for row in index] != paths
                or [int(row["map_index"]) for row in index] != list(range(len(paths)))):
            raise ValueError(f"Map index mismatch: {category}/{engine}")
        values = np.load(prefix.with_suffix(".maps.npy"), mmap_mode="r", allow_pickle=False)
        if values.shape != (len(rows), 256, 256) or values.dtype != np.float32:
            raise ValueError(f"Unexpected map shape or type: {category}/{engine}")
        maps.append(values)

    lo = min(float(values.min()) for values in maps)
    hi = max(float(values.max()) for values in maps)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError(f"Invalid score range: {category}")

    rng = np.random.default_rng(seed)
    weights = np.zeros((repeats + 1, len(rows)), dtype=np.float64)
    weights[0] = 1
    for sample in weights[1:]:
        indices = np.r_[
            rng.choice(normal, len(normal), replace=True),
            rng.choice(anomalous, len(anomalous), replace=True),
        ]
        sample[:] = np.bincount(indices, minlength=len(rows))

    masks, sizes = [], []
    for row, label in zip(rows, labels, strict=True):
        image_path = (data / row["image_path"]).resolve(strict=True)
        if not image_path.is_relative_to(data / category / "test"):
            raise ValueError(f"Unexpected image path: {image_path}")
        with Image.open(image_path) as img:
            width, height = img.size
        if label:
            payload = checked_file(row["mask_path"], row["mask_sha256"],
                                   Path(category) / "ground_truth")
            with Image.open(io.BytesIO(payload)) as img:
                mask = np.asarray(img.convert("L")) > 0
            if mask.shape != (height, width) or not mask.any():
                raise ValueError(f"Invalid anomaly mask: {row['image_path']}")
        else:
            if row["mask_path"] or row["mask_sha256"]:
                raise ValueError(f"Unexpected normal-image mask: {row['image_path']}")
            mask = np.zeros((height, width), dtype=bool)
        masks.append(mask.ravel())
        sizes.append((height, width))

    by_resolution = {fine_bins: [], coarse_bins: []}
    for engine, values in zip(engines, maps, strict=True):
        print(f"BOOTSTRAP={category}/{engine}", flush=True)
        hist = np.zeros((2, len(rows), fine_bins), dtype=np.float64)
        for i, (raw, mask, size) in enumerate(zip(values, masks, sizes, strict=True)):
            resized = torch.nn.functional.interpolate(
                torch.from_numpy(np.array(raw, copy=True))[None, None],
                size=size, mode="bilinear", align_corners=False,
            ).numpy().ravel()
            bucket = ((resized.astype(np.float64) - lo) *
                      (fine_bins / (hi - lo))).astype(np.int32)
            np.clip(bucket, 0, fine_bins - 1, out=bucket)
            hist[0, i] = np.bincount(bucket[~mask], minlength=fine_bins)
            hist[1, i] = np.bincount(bucket[mask], minlength=fine_bins)

        fine = evaluate(hist, weights)
        ratio = fine_bins // coarse_bins
        coarse_hist = hist.reshape(2, len(rows), coarse_bins, ratio).sum(axis=-1)
        coarse = evaluate(coarse_hist, weights)
        by_resolution[fine_bins].append(fine)
        by_resolution[coarse_bins].append(coarse)
        del hist, coarse_hist

    for a, b in contrasts:
        reference, candidate = engines[a], engines[b]
        exact = 100 * (points[category, candidate] - points[category, reference])
        intervals = {}
        estimates = {}
        for bins, results in by_resolution.items():
            delta = 100 * (results[b] - results[a])
            estimates[bins] = delta[0]
            intervals[bins] = np.percentile(delta[1:], [2.5, 97.5], axis=0)
        for j, metric in enumerate(("auroc", "ap")):
            low, high = intervals[fine_bins][:, j]
            shift = float(np.max(np.abs(
                intervals[fine_bins][:, j] - intervals[coarse_bins][:, j]
            )))
            output.append([
                category, reference, candidate, metric, exact[j], low, high,
                abs(estimates[fine_bins][j] - exact[j]), shift,
            ])
    print(f"CATEGORY_COMPLETE={category}", flush=True)

with (out / "pixel_paired_differences.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow([
        "category", "reference", "candidate", "metric", "exact_delta_pp",
        "approx_ci_low_pp", "approx_ci_high_pp", "fine_delta_error_pp",
        "max_ci_endpoint_shift_pp",
    ])
    writer.writerows(output)
print(f"PIXEL_INTERVALS={len(output)} / REPLICATES={repeats} / SEED={seed} "
      f"/ BINS={coarse_bins},{fine_bins}")
