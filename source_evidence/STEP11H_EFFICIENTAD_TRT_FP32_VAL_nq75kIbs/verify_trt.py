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

train, onnx, engines, data, out, trt_site = (
    Path(value).resolve() for value in sys.argv[1:]
)
sys.path.insert(0, str(onnx / "ort_packages"))
sys.path.append(str(trt_site))
import onnxruntime as ort
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
            f"Invalid comparison: TRT {actual.shape}/{actual.dtype}, "
            f"ORT {reference.shape}/{reference.dtype}"
        )
    reference = reference.astype(np.float64)
    error = np.abs(actual.astype(np.float64) - reference)
    return float(error.max()), int(np.count_nonzero(
        error > atol + rtol * np.abs(reference)
    ))

results = {}
for category, count in {"bottle": 21, "carpet": 28}.items():
    print(f"\n=== {category.upper()} TENSORRT FP32 VAL ===", flush=True)
    rows = [
        row for row in manifest
        if row["category"] == category and row["role"] == "VAL"
    ]
    paths = [row["path"] for row in rows]
    with (onnx / f"{category}_onnx_val.csv").open(newline="") as stream:
        previous = list(csv.DictReader(stream))
    if len(set(paths)) != count or paths != [row["path"] for row in previous]:
        raise ValueError(f"{category}: unexpected VAL inventory or order")

    stem = f"{category}_efficientad_s_fp32"
    session = ort.InferenceSession(
        str(onnx / f"{stem}.onnx"), providers=["CPUExecutionProvider"],
    )
    engine = runtime.deserialize_cuda_engine(
        (engines / f"{stem}.engine").read_bytes()
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

    cuda_stream = torch.cuda.current_stream()
    score_max = map_max = 0.0
    score_bad = map_bad = images_bad = 0

    with (out / f"{category}_trt_fp32_val.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "path", "ort_score", "trt_score", "score_abs_error",
            "score_outside_tolerance", "map_max_abs_error",
            "map_pixels_outside_tolerance",
        ])
        for row in rows:
            image = load_input(row, category)
            ort_score, ort_map = session.run(list(output_names), {"image": image.numpy()})

            buffers["image"].copy_(image)
            if not context.execute_async_v3(cuda_stream.cuda_stream):
                raise RuntimeError(f"{category}: inference failed for {row['path']}")
            cuda_stream.synchronize()
            trt_score, trt_map = (buffers[name].cpu().numpy() for name in output_names)

            score_error, bad_score = compare(trt_score, ort_score)
            map_error, bad_map = compare(trt_map, ort_map)
            score_max = max(score_max, score_error)
            map_max = max(map_max, map_error)
            score_bad += bad_score
            map_bad += bad_map
            images_bad += int(bad_score > 0 or bad_map > 0)
            writer.writerow([
                row["path"], ort_score.item(), trt_score.item(),
                score_error, bad_score, map_error, bad_map,
            ])

    results[category] = {
        "val_images": count,
        "score_max_abs_error": score_max,
        "map_max_abs_error": map_max,
        "scores_outside_tolerance": score_bad,
        "pixels_outside_tolerance": map_bad,
        "images_outside_tolerance": images_bad,
        "status": "PASS" if images_bad == 0 else "FAIL",
    }
    print(json.dumps(results[category], indent=2), flush=True)
    del context, engine

passed = all(result["status"] == "PASS" for result in results.values())
summary = {
    "categories": results,
    "tensorrt": trt.__version__,
    "onnxruntime": ort.__version__,
    "reference_provider": "CPUExecutionProvider",
    "torch": torch.__version__,
    "tolerance": {"atol": atol, "rtol": rtol},
    "status": "PASS" if passed else "FAIL",
    "efficientad_test": "NOT_RUN",
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
sys.exit(0 if passed else 2)
