import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

train, fp32, val, fp16, data, out, trt_site = (
    Path(value).resolve() for value in sys.argv[1:]
)
sys.path.append(str(trt_site))
import tensorrt as trt

atol, rtol = 1e-4, 1e-3
output_names = ("pred_score", "anomaly_map")
input_shape = (1, 3, 256, 256)
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

with (train / "input_sha256.csv").open(newline="") as stream:
    manifest = list(csv.DictReader(stream))

def load_input(row, category):
    path = (data / row["path"]).resolve(strict=True)
    if not path.is_relative_to(data / category / "test"):
        raise ValueError(f"Unexpected VAL path: {path}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError(f"VAL input hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        return transform(image.convert("RGB")).unsqueeze(0).contiguous()

def compare(actual, reference):
    if (actual.shape != reference.shape or actual.dtype != np.float32
            or reference.dtype != np.float32
            or not np.isfinite(actual).all()
            or not np.isfinite(reference).all()):
        raise ValueError(
            f"Invalid comparison: FP16 candidate {actual.shape}/{actual.dtype}, "
            f"FP32 reference {reference.shape}/{reference.dtype}"
        )
    reference = reference.astype(np.float64)
    error = np.abs(actual.astype(np.float64) - reference)
    return float(error.max()), int(np.count_nonzero(
        error > atol + rtol * np.abs(reference)
    ))

def open_engine(engine_path, category):
    engine = runtime.deserialize_cuda_engine(
        engine_path.read_bytes()
    )
    if engine is None:
        raise RuntimeError(f"{category}: could not deserialize engine")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"{category}: could not create execution context")

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    if len(names) != 3 or set(names) != {"image", *output_names}:
        raise ValueError(f"{category}: unexpected tensor names {names}")

    buffers = {}
    for name in names:
        mode = trt.TensorIOMode.INPUT if name == "image" else trt.TensorIOMode.OUTPUT
        if (engine.get_tensor_mode(name) != mode
                or engine.get_tensor_dtype(name) != trt.float32
                or engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
                or engine.get_tensor_location(name) != trt.TensorLocation.DEVICE):
            raise ValueError(f"{category}: unexpected tensor interface for {name}")
        shape = tuple(context.get_tensor_shape(name))
        if any(d <= 0 for d in shape):
            raise ValueError(f"{category}: unresolved tensor shape {name}: {shape}")
        buffers[name] = torch.empty(shape, device="cuda", dtype=torch.float32)
        if not context.set_tensor_address(name, buffers[name].data_ptr()):
            raise RuntimeError(f"{category}: buffer binding failed for {name}")

    if (tuple(buffers["image"].shape) != input_shape
            or buffers["pred_score"].numel() != 1
            or tuple(buffers["anomaly_map"].shape) != (1, 1, 256, 256)):
        raise ValueError(f"{category}: unexpected tensor shapes")

    return engine, context, buffers

def infer(context, buffers, image, cuda_stream):
    with torch.cuda.stream(cuda_stream):
        buffers["image"].copy_(image)
        if not context.execute_async_v3(cuda_stream.cuda_stream):
            raise RuntimeError("TensorRT inference failed")
    cuda_stream.synchronize()
    return tuple(buffers[name].cpu().numpy() for name in output_names)



def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, records):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def counts(labels, flags):
    return {
        "tp": int(np.sum(labels & flags)),
        "tn": int(np.sum(~labels & ~flags)),
        "fp": int(np.sum(~labels & flags)),
        "fn": int(np.sum(labels & ~flags)),
    }


def kendall_tau_b(a, b):
    i, j = np.triu_indices(len(a), k=1)
    da = np.sign(a[i] - a[j])
    db = np.sign(b[i] - b[j])
    denominator = np.sqrt(float(np.count_nonzero(da)) * np.count_nonzero(db))
    return float(np.sum(da * db) / denominator) if denominator else None


def shared_threshold_peak(a, b):
    thresholds = np.unique(np.r_[a, b])
    counts_at_threshold = np.count_nonzero(
        (a[None, :] > thresholds[:, None]) != (b[None, :] > thresholds[:, None]),
        axis=1,
    )
    i = int(np.argmax(counts_at_threshold))
    return int(counts_at_threshold[i]), float(thresholds[i])


test_rows = read_csv(fp16 / "test_manifest.csv")
historical = read_csv(fp16 / "test_image_scores.csv")
inventory = json.loads((val / "summary.json").read_text())["categories"]
results, decisions, per_image = [], [], []

for category, (val_count, test_count, normal_count) in {
    "bottle": (21, 83, 20),
    "carpet": (28, 117, 28),
}.items():
    rows = [r for r in test_rows if r["category"] == category]
    paths = [r["image_path"] for r in rows]
    if any(r["anomalous"] not in {"0", "1"} for r in rows):
        raise ValueError(f"{category}: unexpected test label")
    labels = np.array([r["anomalous"] == "1" for r in rows])
    if len(paths) != test_count or len(set(paths)) != test_count or sum(~labels) != normal_count:
        raise ValueError(f"{category}: unexpected test inventory")

    configurations = inventory[category]["engines"]
    runners = {
        name: open_engine(Path(path), f"{category}/{name}")
        for name, path in configurations.items()
    }
    stream = torch.cuda.Stream()
    scores = {name: [] for name in configurations}
    for row in rows:
        path = Path(row["image_path"])
        if (path.parts[:2] != (category, "test") or len(path.parts) < 4
                or (path.parts[2] == "good") != (row["anomalous"] == "0")):
            raise ValueError(f"Unexpected path and label: {path}")
        image = load_input({
            "path": row["image_path"],
            "sha256": row["image_sha256"],
        }, category)
        record = {"category": category, "path": row["image_path"], "label": row["anomalous"]}
        for name, (_, context, buffers) in runners.items():
            score, _ = infer(context, buffers, image, stream)
            value = float(score.item())
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite score: {category}/{name}/{path}")
            scores[name].append(value)
            record[name] = value
        per_image.append(record)
    scores = {name: np.asarray(values) for name, values in scores.items()}

    control_rows = [
        r for r in historical
        if r["category"] == category and r["configuration"] == "automatic_fp16_enabled"
    ]
    control = {r["image_path"]: r for r in control_rows}
    if len(control_rows) != test_count or set(control) != set(paths):
        raise ValueError(f"{category}: saved test control inventory differs")
    for name, column in (
        ("original_no_tf32", "fp32_score"),
        ("automatic_fp16", "candidate_score"),
    ):
        expected = np.array([float(control[path][column]) for path in paths])
        delta = np.abs(scores[name] - expected)
        if not np.isfinite(delta).all():
            raise ValueError(f"{category}/{name}: invalid saved control scores")
        print(f"TEST_SCORE_REPLAY={category}/{name} MAX_ABS={delta.max():.9g}", flush=True)
        if not np.all(delta <= 1e-6):
            raise ValueError(f"{category}/{name}: saved test control was not reproduced")

    val_scores = {name: {} for name in configurations}
    for row in read_csv(val / f"{category}_per_image.csv"):
        for side in ("reference", "candidate"):
            name, path = row[side], row["path"]
            score = float(row[f"{side}_score"])
            previous = val_scores[name].setdefault(path, score)
            if not np.isfinite(score) or previous != score:
                raise ValueError(f"Inconsistent VAL score: {category}/{name}/{path}")
    val_sets = [set(values) for values in val_scores.values()]
    if len(val_sets[0]) != val_count or any(s != val_sets[0] for s in val_sets[1:]):
        raise ValueError(f"{category}: VAL inventory differs across engines")

    comparisons = [
        ("original_no_tf32", "default_tf32"),
        ("default_tf32", "automatic_fp16"),
    ]
    if category == "carpet":
        comparisons.append(("default_tf32", "decoder_conv_fp16"))

    for reference, candidate in comparisons:
        a, b = scores[reference], scores[candidate]
        peak, peak_threshold = shared_threshold_peak(a, b)
        i = int(np.argmax(np.abs(b - a)))
        result = {
            "category": category,
            "reference": reference,
            "candidate": candidate,
            "test_images": test_count,
            "kendall_tau_b": kendall_tau_b(a, b),
            "max_abs_score_delta": float(abs(b[i] - a[i])),
            "max_delta_image": paths[i],
            "reference_score_at_max_delta": float(a[i]),
            "candidate_score_at_max_delta": float(b[i]),
            "peak_shared_threshold_disagreements": peak,
            "threshold_at_peak": peak_threshold,
        }
        results.append(result)
        print(json.dumps(result), flush=True)

        for point, q in (("val_max", 1.0), ("val_p95", 0.95)):
            ta = float(np.quantile(list(val_scores[reference].values()), q))
            tb = float(np.quantile(list(val_scores[candidate].values()), q))
            for policy, used_ta in (("own", ta), ("shared_candidate", tb)):
                fa, fb = a > used_ta, b > tb
                ca, cb = counts(labels, fa), counts(labels, fb)
                decision = {
                    "category": category, "reference": reference, "candidate": candidate,
                    "operating_point": point, "policy": policy,
                    "reference_threshold": used_ta, "candidate_threshold": tb,
                    "disagreements": int(np.count_nonzero(fa != fb)),
                    **{f"reference_{k}": v for k, v in ca.items()},
                    **{f"candidate_{k}": v for k, v in cb.items()},
                }
                decisions.append(decision)
                print(
                    f"DECISIONS={category}/{reference}->{candidate}/{point}/{policy} "
                    f"D={decision['disagreements']}/{test_count} "
                    f"FP={ca['fp']}->{cb['fp']} FN={ca['fn']}->{cb['fn']}",
                    flush=True,
                )
    del runners

if len(results) != 5 or len(decisions) != 20 or len(per_image) != 200:
    raise ValueError("Incomplete result inventory")
# Bottle has no decoder column. DictWriter uses a shared schema across categories.
fields = list(dict.fromkeys(key for row in per_image for key in row))
with (out / "test_scores.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(per_image)
write_csv(out / "test_comparisons.csv", results)
write_csv(out / "operating_points.csv", decisions)
print("COMPARISONS=5 / OPERATING_POINT_ROWS=20 / UNIQUE_TEST_IMAGES=200")
