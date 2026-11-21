import ast
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

audit, build, out, trt_site = map(Path, sys.argv[1:])
sys.path.append(str(trt_site))
import tensorrt as trt

data = audit.parent / "04_Datasets/MVTec_AD/01_Extracted"
train = audit / "STEP11E_EFFICIENTAD_TRAIN_ZeV4ionZ"
references = audit / "STEP13A_R1_EFFICIENTAD_TF32_REFERENCES_G2Jd0c1J"
source = audit / "STEP11J_EFFICIENTAD_TRT_FP16_VAL_kygZcMS9/verify_fp16.py"

atol, rtol = 1e-4, 1e-3
input_shape = (1, 3, 256, 256)
output_names = ("pred_score", "anomaly_map")
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

tree = ast.parse(source.read_text())
functions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in {"open_engine", "infer", "compare"}
]
if {node.name for node in functions} != {"open_engine", "infer", "compare"}:
    raise ValueError("Expected existing engine, inference and comparison functions")
exec(compile(
    ast.Module(body=functions, type_ignores=[]), str(source), "exec"
))

with (train / "input_sha256.csv").open(newline="") as stream:
    manifest = list(csv.DictReader(stream))
jobs = json.loads((build / "summary.json").read_text())["completed"]
if len(jobs) != 12:
    raise ValueError("Expected 12 STEP16A engines")

protocol = {
    "reference": "strongly typed TF32, same typing mode as candidates",
    "candidate": "strongly typed mixed ONNX, TF32 allowed",
    "tolerance": {"atol": atol, "rtol": rtol},
    "scope": "normal VAL",
    "source_functions": str(source),
    "tensorrt": trt.__version__,
    "torch": torch.__version__,
}
(out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

results = []
tactics = []
for category, expected in {"bottle": 21, "carpet": 28}.items():
    rows = [
        row for row in manifest
        if row["category"] == category and row["role"] == "VAL"
    ]
    if len(rows) != expected or len({r["path"] for r in rows}) != expected:
        raise ValueError(f"Unexpected VAL inventory: {category}")

    images = []
    for row in rows:
        path = (data / row["path"]).resolve(strict=True)
        if not path.is_relative_to((data / category / "train/good").resolve()):
            raise ValueError(f"Unexpected VAL path: {path}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"Input hash mismatch: {path}")
        with Image.open(io.BytesIO(payload)) as image:
            images.append(
                transform(image.convert("RGB")).unsqueeze(0).contiguous()
            )

    ref_path = references / category / "strongly_typed_tf32_allowed/model.engine"
    ref_engine, ref_context, ref_buffers = open_engine(ref_path, category)
    cuda_stream = torch.cuda.Stream()
    ref_outputs = [
        infer(ref_context, ref_buffers, image, cuda_stream)
        for image in images
    ]

    for job in [j for j in jobs if j["category"] == category]:
        configuration = job["configuration"]
        engine_path = build / job["engine"]
        if hashlib.sha256(engine_path.read_bytes()).hexdigest() != job["engine_sha256"]:
            raise ValueError(f"Engine hash mismatch: {engine_path}")

        engine, context, buffers = open_engine(engine_path, category)
        score_max = map_max = 0.0
        score_bad = map_bad = images_bad = 0
        path = out / f"{category}_{configuration}_val.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "path", "reference_score", "candidate_score",
                "score_abs_error", "score_outside_tolerance",
                "map_max_abs_error", "map_pixels_outside_tolerance",
            ])
            for row, image, (ref_score, ref_map) in zip(
                rows, images, ref_outputs, strict=True
            ):
                score, anomaly_map = infer(
                    context, buffers, image, cuda_stream
                )
                score_error, bad_score = compare(score, ref_score)
                map_error, bad_map = compare(anomaly_map, ref_map)
                score_max = max(score_max, score_error)
                map_max = max(map_max, map_error)
                score_bad += bad_score
                map_bad += bad_map
                images_bad += int(bad_score > 0 or bad_map > 0)
                writer.writerow([
                    row["path"], ref_score.item(), score.item(),
                    score_error, bad_score, map_error, bad_map,
                ])

        result = {
            "category": category,
            "configuration": configuration,
            "val_images": expected,
            "score_max_abs_error": score_max,
            "map_max_abs_error": map_max,
            "scores_outside_tolerance": score_bad,
            "pixels_outside_tolerance": map_bad,
            "images_outside_tolerance": images_bad,
            "status": "PASS" if images_bad == 0 else "FAIL",
        }
        results.append(result)
        print(json.dumps(result), flush=True)

        info = json.loads(
            (engine_path.parent / "layer_info.json").read_text()
        )
        layers = info.get("Layers", []) if isinstance(info, dict) else info
        for layer in layers:
            if isinstance(layer, dict):
                tactics.append({
                    "category": category,
                    "configuration": configuration,
                    "name": layer.get("Name", ""),
                    "type": layer.get("LayerType", ""),
                    "tactic": layer.get("TacticName", ""),
                    "inputs": layer.get("Inputs", []),
                    "outputs": layer.get("Outputs", []),
                })
        del context, buffers, engine

    del ref_context, ref_buffers, ref_engine

report = {
    **protocol,
    "comparisons": results,
    "comparisons_complete": len(results),
    "efficientad_test": "NOT_RUN",
}
(out / "summary.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n"
)
(out / "layer_tactics.json").write_text(
    json.dumps(tactics, indent=2) + "\n"
)
print(f"VAL_COMPARISONS_COMPLETE={len(results)}/12", flush=True)
