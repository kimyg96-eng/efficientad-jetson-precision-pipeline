import ast
import csv
import json
import sys
from contextlib import ExitStack
from pathlib import Path

train, build, prior, data, out = (
    Path(value).resolve(strict=True) for value in sys.argv[1:]
)
sys.path.append('/usr/lib/python3.12/dist-packages')

import hashlib
import io

import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from torchvision import transforms

if trt.__version__ != "10.16.2.10":
    raise RuntimeError(f"Unexpected TensorRT version: {trt.__version__}")

atol, rtol = 1e-4, 1e-3
input_shape = (1, 3, 256, 256)
output_names = ("pred_score", "anomaly_map")
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

source = prior / "verify_fp16.py"
tree = ast.parse(source.read_text(), filename=str(source))
names = {"load_input", "open_engine", "infer", "compare"}
functions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in names
]
if len(functions) != len(names) or {node.name for node in functions} != names:
    raise RuntimeError("Unexpected STEP11J helper functions")
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"))

configs = ("strict_fp32", "default_tf32", "automatic_fp16")
contrasts = (
    ("strict_fp32", "default_tf32"),
    ("default_tf32", "automatic_fp16"),
)

def validate_outputs(score, anomaly_map):
    for name, value in zip(output_names, (score, anomaly_map), strict=True):
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name} output")

def evaluate(category, count):
    category_dir = train / category
    with (category_dir / "input_sha256.csv").open(newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["category"] == category and row["role"] == "VAL"
        ]
    with (category_dir / f"{category}_val_scores.csv").open(newline="") as stream:
        saved_paths = [row["path"] for row in csv.DictReader(stream)]
    paths = [row["path"] for row in rows]
    if len(set(paths)) != count or paths != saved_paths:
        raise ValueError(f"{category}: unexpected VAL inventory or order")

    engines = {
        config: open_engine(build / category / config / "model.engine", category)
        for config in configs
    }
    stream = torch.cuda.Stream()
    scores = {config: [] for config in configs}
    comparisons = {}

    with ExitStack() as stack:
        score_writer = csv.writer(stack.enter_context(
            (out / f"{category}_val_scores.csv").open("w", newline="")
        ))
        score_writer.writerow(["path", *configs])
        writers = {}
        for reference, candidate in contrasts:
            label = f"{reference}_vs_{candidate}"
            writers[label] = csv.writer(stack.enter_context(
                (out / f"{category}_{label}.csv").open("w", newline="")
            ))
            writers[label].writerow([
                "path", "score_abs_error", "score_outside_tolerance",
                "map_max_abs_error", "map_pixels_outside_tolerance",
            ])
            comparisons[label] = {
                "val_images": count,
                "score_max_abs_error": 0.0, "map_max_abs_error": 0.0,
                "scores_outside_tolerance": 0, "pixels_outside_tolerance": 0,
                "images_outside_tolerance": 0,
            }

        for row in rows:
            image = load_input(row, category)
            outputs = {}
            for config, (_, context, buffers) in engines.items():
                outputs[config] = infer(context, buffers, image, stream)
                validate_outputs(*outputs[config])
                scores[config].append(outputs[config][0].item())
            score_writer.writerow([row["path"], *(scores[c][-1] for c in configs)])

            for reference, candidate in contrasts:
                label = f"{reference}_vs_{candidate}"
                ref_score, ref_map = outputs[reference]
                cand_score, cand_map = outputs[candidate]
                score_error, score_bad = compare(cand_score, ref_score)
                map_error, map_bad = compare(cand_map, ref_map)
                writers[label].writerow([
                    row["path"], score_error, score_bad, map_error, map_bad,
                ])
                result = comparisons[label]
                result["score_max_abs_error"] = max(result["score_max_abs_error"], score_error)
                result["map_max_abs_error"] = max(result["map_max_abs_error"], map_error)
                result["scores_outside_tolerance"] += score_bad
                result["pixels_outside_tolerance"] += map_bad
                result["images_outside_tolerance"] += int(bool(score_bad or map_bad))

    for result in comparisons.values():
        result["status"] = "FAIL" if result["images_outside_tolerance"] else "PASS"

    thresholds = {
        config: {
            "val_max": max(values),
            "val_p95_linear": float(np.quantile(values, 0.95, method="linear")),
        }
        for config, values in scores.items()
    }
    return {"comparisons": comparisons, "thresholds": thresholds}

report = {
    "tensorrt": trt.__version__,
    "torch": torch.__version__,
    "tolerance": {"atol": atol, "rtol": rtol},
    "decision_rule": "score > threshold",
    "scope": "normal VAL only",
    "efficientad_test": "NOT_RUN",
    "categories": {},
}
for category, count in {"capsule": 22, "grid": 27}.items():
    report["categories"][category] = evaluate(category, count)

text = json.dumps(report, indent=2)
(out / "summary.json").write_text(text + "\n")
print(text, flush=True)
