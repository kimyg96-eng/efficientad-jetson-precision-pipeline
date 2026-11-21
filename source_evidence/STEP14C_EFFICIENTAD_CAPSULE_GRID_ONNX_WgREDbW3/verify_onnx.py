import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from PIL import Image
from torchvision import transforms
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel

src, data, out, ort_dir = (Path(arg).resolve(strict=True) for arg in sys.argv[1:])
sys.path.insert(0, str(ort_dir))
import onnxruntime as ort

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

categories = {"capsule": 22, "grid": 27}
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        result = self.model(image)
        return result.pred_score, result.anomaly_map

def load_image(row, category):
    path = (data / row["path"]).resolve(strict=True)
    if not path.is_relative_to(data / category / "train" / "good"):
        raise ValueError(f"Unexpected VAL path: {path}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError(f"VAL image SHA-256 mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        return transform(image.convert("RGB")).unsqueeze(0)

def compare(actual, reference, atol, rtol):
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise ValueError(f"Invalid output shape or values: {actual.shape}")
    error = np.abs(actual.astype(np.float64) - reference)
    return float(error.max()), int(np.count_nonzero(
        error > atol + rtol * np.abs(reference)
    ))

summary = {}
for category, count in categories.items():
    category_src = src / category
    with (category_src / "input_sha256.csv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    print(f"\n=== {category.upper()} CHECKPOINT AND ONNX ===", flush=True)
    rows = [
        row for row in manifest
        if row["category"] == category and row["role"] == "VAL"
    ]
    with (category_src / f"{category}_val_scores.csv").open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    paths = [row["path"] for row in rows]
    if (len(rows) != count or len(set(paths)) != count
            or paths != [row["path"] for row in saved]):
        raise ValueError(f"{category}: unexpected VAL image count or order")

    model = EfficientAdModel()
    state = torch.load(
        category_src / f"{category}_efficientad_s_fp32.pt",
        map_location="cpu", weights_only=True,
    )

    # Register the learned statistics before strict checkpoint restoration.
    for name, keys in (
        ("mean_std", ("mean", "std")),
        ("quantiles", ("qa_st", "qb_st", "qa_ae", "qb_ae")),
    ):
        getattr(model, name).update({
            key: torch.empty_like(state[f"{name}.{key}"]) for key in keys
        })

    model.load_state_dict(state, strict=True)
    model = ExportModel(model).cuda().eval()
    inputs = [load_image(row, category) for row in rows]

    onnx_path = out / f"{category}_efficientad_s_fp32.onnx"
    torch.onnx.export(
        model, (inputs[0].cuda(),), str(onnx_path),
        input_names=["image"],
        output_names=["pred_score", "anomaly_map"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(onnx_path))
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"],
    )

    restored_max = score_max = map_max = 0.0
    restore_bad = score_bad = map_bad = 0
    with (out / f"{category}_onnx_val.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "path", "step14b_score", "restored_score", "onnx_score",
            "onnx_map_max_abs_error", "onnx_map_outside_tolerance",
        ])
        for row, prior, image in zip(rows, saved, inputs, strict=True):
            with torch.inference_mode():
                score, anomaly_map = (v.cpu().numpy() for v in model(image.cuda()))
            if (score.size != 1 or anomaly_map.shape != (1, 1, 256, 256)
                    or not np.isfinite(score).all()
                    or not np.isfinite(anomaly_map).all()):
                raise ValueError(f"{category}: invalid restored PyTorch output")

            prior_score = float(prior["score"])
            restored_error, bad_restore = compare(
                score, np.full_like(score, prior_score), atol=1e-6, rtol=1e-5,
            )

            ort_score, ort_map = session.run(
                ["pred_score", "anomaly_map"], {"image": image.numpy()},
            )
            score_error, bad_score = compare(
                ort_score, score, atol=1e-4, rtol=1e-3,
            )
            map_error, bad_map = compare(
                ort_map, anomaly_map, atol=1e-4, rtol=1e-3,
            )

            restored_max = max(restored_max, restored_error)
            score_max = max(score_max, score_error)
            map_max = max(map_max, map_error)
            restore_bad += bad_restore
            score_bad += bad_score
            map_bad += bad_map
            writer.writerow([
                row["path"], prior_score, score.item(), ort_score.item(),
                map_error, bad_map,
            ])

    passed = restore_bad == score_bad == map_bad == 0
    summary[category] = {
        "val_images": count,
        "restored_score_max_abs_error": restored_max,
        "restored_scores_outside_tolerance": restore_bad,
        "onnx_score_max_abs_error": score_max,
        "onnx_scores_outside_tolerance": score_bad,
        "onnx_map_max_abs_error": map_max,
        "onnx_pixels_outside_tolerance": map_bad,
        "status": "PASS" if passed else "FAIL",
    }
    print(json.dumps(summary[category], indent=2), flush=True)

passed = all(item["status"] == "PASS" for item in summary.values())
report = {
    "categories": summary,
    "torch": torch.__version__,
    "onnx": onnx.__version__,
    "onnxruntime": ort.__version__,
    "onnx_opset": 17,
    "onnxruntime_provider": "CPUExecutionProvider",
    "checkpoint_score_tolerance": {"atol": 1e-6, "rtol": 1e-5},
    "onnx_output_tolerance": {"atol": 1e-4, "rtol": 1e-3},
    "status": "PASS" if passed else "FAIL",
    "efficientad_test": "NOT_RUN",
}
(out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
if not passed:
    sys.exit(2)
