import ast
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import transforms

fp32, split, data, out, trt_site = (Path(arg).resolve() for arg in sys.argv[1:])
sys.path.append(str(trt_site))
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
input_shape = (1, 3, 256, 256)
output_names = ("pred_score", "anomaly_map")
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Reuse the engine interface checks that passed STEP11N and STEP11P.
source = split / "verify_conv_fp16.py"
tree = ast.parse(source.read_text())
functions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "open_engine"
]
if len(functions) != 1:
    raise ValueError("Expected one STEP11N open_engine function")
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"))

with (out / "test_manifest.csv").open(newline="") as stream:
    manifest = list(csv.DictReader(stream))

def checked_bytes(relative, expected, subtree):
    path = (data / relative).resolve(strict=True)
    if not path.is_relative_to(data / subtree):
        raise ValueError(f"Unexpected test path: {relative}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {relative}")
    return payload

def metrics(labels, scores):
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
    }

results = {}
for category, expected in {"bottle": (20, 63), "carpet": (28, 89)}.items():
    rows = [row for row in manifest if row["category"] == category]
    labels = np.array([int(row["anomalous"]) for row in rows], dtype=np.uint8)
    counts = (int((labels == 0).sum()), int((labels == 1).sum()))
    if (counts != expected or len(rows) != sum(expected)
            or len({row["image_path"] for row in rows}) != len(rows)):
        raise ValueError(f"{category}: unexpected frozen test inventory")

    images, masks, sizes = [], [], []
    for row, label in zip(rows, labels, strict=True):
        payload = checked_bytes(
            row["image_path"], row["image_sha256"], f"{category}/test",
        )
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            images.append(transform(image.convert("RGB")).unsqueeze(0).contiguous())
        sizes.append((height, width))

        if label:
            payload = checked_bytes(
                row["mask_path"], row["mask_sha256"], f"{category}/ground_truth",
            )
            with Image.open(io.BytesIO(payload)) as mask_image:
                mask = (np.asarray(mask_image.convert("L")) > 0).astype(np.uint8)
            if mask.shape != (height, width) or not mask.any():
                raise ValueError(f"Invalid anomaly mask: {row['mask_path']}")
        else:
            if row["mask_path"] or row["mask_sha256"]:
                raise ValueError(f"Unexpected normal-image mask: {row['image_path']}")
            mask = np.zeros((height, width), dtype=np.uint8)
        masks.append(mask.ravel())

    pixel_labels = np.concatenate(masks)
    engines = {"fp32": fp32 / f"{category}_efficientad_s_fp32.engine"}
    if category == "carpet":
        engines["decoder_conv_fp16"] = (
            split / "decoder/carpet_efficientad_s_conv_fp16.engine"
        )

    results[category] = {
        "test_images": len(rows),
        "normal_images": counts[0],
        "anomalous_images": counts[1],
        "pixels": int(pixel_labels.size),
        "positive_pixels": int(np.count_nonzero(pixel_labels)),
        "engines": {},
    }

    for name, path in engines.items():
        print(f"RUN={category}/{name} IMAGES={len(rows)}", flush=True)
        engine, context, buffers = open_engine(path, category)
        cuda_stream = torch.cuda.Stream()
        raw_maps = np.empty((len(rows), 256, 256), dtype=np.float32)
        scores = np.empty(len(rows), dtype=np.float32)

        for i, image in enumerate(images):
            with torch.cuda.stream(cuda_stream):
                buffers["image"].copy_(image)
                if not context.execute_async_v3(cuda_stream.cuda_stream):
                    raise RuntimeError(f"Inference failed: {rows[i]['image_path']}")
            cuda_stream.synchronize()
            scores[i] = buffers["pred_score"].item()
            raw_maps[i] = buffers["anomaly_map"].cpu().numpy()[0, 0]

        if not np.isfinite(scores).all() or not np.isfinite(raw_maps).all():
            raise ValueError(f"Non-finite inference output: {category}/{name}")

        prefix = out / f"{category}_{name}"
        np.save(prefix.with_suffix(".maps.npy"), raw_maps, allow_pickle=False)
        with prefix.with_suffix(".scores.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["map_index", "image_path", "defect", "anomalous", "pred_score"])
            for i, (row, score) in enumerate(zip(rows, scores, strict=True)):
                writer.writerow([i, row["image_path"], row["defect"],
                                 row["anomalous"], float(score)])

        pixel_scores = np.empty(pixel_labels.size, dtype=np.float32)
        offset = 0
        for raw_map, (height, width) in zip(raw_maps, sizes, strict=True):
            resized = torch.nn.functional.interpolate(
                torch.from_numpy(raw_map)[None, None],
                size=(height, width), mode="bilinear", align_corners=False,
            ).numpy().ravel()
            pixel_scores[offset:offset + resized.size] = resized
            offset += resized.size

        result = {
            "image": metrics(labels, scores),
            "pixel": metrics(pixel_labels, pixel_scores),
        }
        results[category]["engines"][name] = result
        print(json.dumps(result), flush=True)
        del pixel_scores, context, engine, buffers

carpet = results["carpet"]["engines"]
deltas = {
    level: {
        metric: 100 * (
            carpet["decoder_conv_fp16"][level][metric] - carpet["fp32"][level][metric]
        )
        for metric in ("auroc", "ap")
    }
    for level in ("image", "pixel")
}
report = {
    "categories": results,
    "carpet_decoder_fp16_minus_fp32_percentage_points": deltas,
    "pixel_protocol": "native-resolution bilinear interpolation, align_corners=False; pooled by category",
    "versions": {"tensorrt": trt.__version__, "torch": torch.__version__,
                 "scikit_learn": sklearn.__version__},
    "efficientad_test": "COMPLETE",
}
text = json.dumps(report, indent=2, allow_nan=False)
(out / "summary.json").write_text(text + "\n")
print(text)
