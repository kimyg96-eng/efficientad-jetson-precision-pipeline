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
    if not path.is_relative_to(data / category / "train" / "good"):
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


import os

refs = Path(os.environ["STEP13G_REFS"])
split = Path(os.environ["STEP13G_SPLIT"])
results = {}

for category, count in {"bottle": 21, "carpet": 28}.items():
    rows = [
        row for row in manifest
        if row["category"] == category and row["role"] == "VAL"
    ]
    paths = [row["path"] for row in rows]
    with (val / f"{category}_trt_fp32_val.csv").open(newline="") as stream:
        previous = list(csv.DictReader(stream))
    if len(paths) != count or len(set(paths)) != count or paths != [
        row["path"] for row in previous
    ]:
        raise ValueError(f"{category}: unexpected frozen VAL inventory or order")

    stem = f"{category}_efficientad_s"
    engines = {
        "original_no_tf32": fp32 / f"{stem}_fp32.engine",
        "default_tf32": refs / category / "default_tf32_allowed/model.engine",
        "automatic_fp16": fp16 / f"{stem}_fp16.engine",
    }
    comparisons = [
        ("original_no_tf32", "default_tf32"),
        ("default_tf32", "automatic_fp16"),
    ]
    if category == "carpet":
        engines["decoder_conv_fp16"] = split / "decoder/carpet_efficientad_s_conv_fp16.engine"
        comparisons.append(("default_tf32", "decoder_conv_fp16"))

    runners = {
        name: open_engine(path, f"{category}/{name}")
        for name, path in engines.items()
    }
    stream = torch.cuda.Stream()
    totals = {
        f"{reference}_vs_{candidate}": {
            "val_images": count,
            "score_max_abs_error": 0.0,
            "map_max_abs_error": 0.0,
            "scores_outside_tolerance": 0,
            "pixels_outside_tolerance": 0,
            "images_outside_tolerance": 0,
        }
        for reference, candidate in comparisons
    }

    with (out / f"{category}_per_image.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "path", "reference", "candidate", "reference_score", "candidate_score",
            "score_abs_error", "score_outside_tolerance", "map_max_abs_error",
            "map_pixels_outside_tolerance",
        ])
        for row in rows:
            image = load_input(row, category)
            outputs = {
                name: infer(context, buffers, image, stream)
                for name, (_, context, buffers) in runners.items()
            }
            for reference, candidate in comparisons:
                ref_score, ref_map = outputs[reference]
                score, anomaly_map = outputs[candidate]
                score_error, bad_score = compare(score, ref_score)
                map_error, bad_map = compare(anomaly_map, ref_map)
                total = totals[f"{reference}_vs_{candidate}"]
                total["score_max_abs_error"] = max(total["score_max_abs_error"], score_error)
                total["map_max_abs_error"] = max(total["map_max_abs_error"], map_error)
                total["scores_outside_tolerance"] += bad_score
                total["pixels_outside_tolerance"] += bad_map
                total["images_outside_tolerance"] += int(bad_score > 0 or bad_map > 0)
                writer.writerow([
                    row["path"], reference, candidate, ref_score.item(), score.item(),
                    score_error, bad_score, map_error, bad_map,
                ])

    for total in totals.values():
        total["status"] = "PASS" if total["images_outside_tolerance"] == 0 else "FAIL"
    results[category] = {
        "engines": {name: str(path) for name, path in engines.items()},
        "comparisons": totals,
    }
    del runners

summary = {
    "categories": results,
    "tensorrt": trt.__version__,
    "torch": torch.__version__,
    "tolerance": {"atol": atol, "rtol": rtol},
    "scope": "normal VAL only",
}
report = json.dumps(summary, indent=2)
(out / "summary.json").write_text(report + "\n")
print(report)
