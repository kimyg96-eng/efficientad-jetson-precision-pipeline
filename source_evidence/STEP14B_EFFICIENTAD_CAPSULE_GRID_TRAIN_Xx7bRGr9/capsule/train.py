import csv
import hashlib
import io
import json
import math
import random
import sys
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from anomalib.models.image.efficient_ad.lightning_model import EfficientAd
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize

data, manifest, assets, smoke, out = (Path(x).resolve() for x in sys.argv[1:])
seed = 20261116
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def digest(payload):
    return hashlib.sha256(payload).hexdigest()

smoke_info = json.loads((smoke / "summary.json").read_text())
with zipfile.ZipFile(assets / "efficientad_pretrained_weights.zip") as archive:
    teacher_bytes = archive.read(smoke_info["teacher_zip_member"])
if digest(teacher_bytes) != smoke_info["teacher_sha256"]:
    raise SystemExit("Teacher SHA-256 mismatch")
teacher = torch.load(io.BytesIO(teacher_bytes), map_location="cpu", weights_only=True)

steps = 40000
labels_root = out / "imagenette"
with tarfile.open(assets / "imagenette2.tgz", "r|gz") as archive:
    for member in archive:
        parts = PurePosixPath(member.name).parts
        if (member.isfile() and len(parts) == 4 and parts[1] == "train"
                and PurePosixPath(member.name).suffix.lower() in {".jpeg", ".jpg"}):
            if any(part in {"", ".", "..", "/"} for part in parts):
                raise ValueError(f"Invalid ImageNette archive path: {member.name}")
            archive.extract(member, path=labels_root, filter="data")

roots = sorted(path for path in labels_root.glob("*/train") if path.is_dir())
if len(roots) != 1:
    raise SystemExit("Expected one extracted ImageNette train directory")

normal_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
auxiliary_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
])

with open(manifest, newline="") as stream:
    rows = list(csv.DictReader(stream))

def tensor_from_bytes(payload, transform):
    with Image.open(io.BytesIO(payload)) as image:
        return transform(image.convert("RGB")).unsqueeze(0)

used_inputs = []
def read_normal(row, category, role):
    path = (data / row["path"]).resolve(strict=True)
    if not path.is_relative_to(data / category / "train" / "good"):
        raise ValueError(f"Unexpected normal image path: {path}")
    payload = path.read_bytes()
    sha256 = digest(payload)
    expected = row.get("sha256", "").strip().lower()
    if expected and sha256 != expected:
        raise ValueError(f"Image SHA-256 mismatch: {path}")
    used_inputs.append({
        "category": category, "role": role, "path": row["path"], "sha256": sha256,
    })
    return tensor_from_bytes(payload, normal_transform)

auxiliary_dataset = ImageFolder(roots[0], transform=auxiliary_transform)
if len(auxiliary_dataset.classes) != 10 or not auxiliary_dataset.samples:
    raise ValueError("Unexpected ImageNette class inventory")
print(f"IMAGENETTE_TRAIN_IMAGES={len(auxiliary_dataset)}", flush=True)
results = {}
for category, expected in {'capsule': (131, 22)}.items():
    train, val = (
        sorted(
            (row for row in rows
             if row["category"].strip().lower() == category
             and row["role"].strip().upper() == role),
            key=lambda row: row["path"],
        )
        for role in ("TRAIN", "VAL")
    )
    if (len(train), len(val)) != expected:
        raise ValueError(f"{category}: unexpected split sizes")
    if len({row["path"] for row in train + val}) != sum(expected):
        raise ValueError(f"{category}: duplicate TRAIN/VAL paths")

    random.seed(seed)
    torch.manual_seed(seed)
    module = EfficientAd(model_size=EfficientAdModelSize.S).cuda()
    model = module.model
    model.teacher.load_state_dict(teacher, strict=True)
    model.teacher.requires_grad_(False).eval()

    train_tensors = [read_normal(row, category, "TRAIN") for row in train]
    val_tensors = [read_normal(row, category, "VAL") for row in val]

    def teacher_batches():
        for tensor in train_tensors:
            yield SimpleNamespace(image=tensor)

    stats = module.teacher_channel_mean_std(teacher_batches())
    if (any(not torch.isfinite(v).all().item() for v in stats.values())
            or not (stats["std"] > 0).all().item()):
        raise ValueError(f"{category}: invalid teacher statistics")
    model.mean_std.update(stats)

    model.train()
    model.teacher.eval()
    params = [*model.student.parameters(), *model.ae.parameters()]
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * steps), gamma=0.1,
    )

    def shuffled_indices(size, generator):
        while True:
            yield from torch.randperm(size, generator=generator).tolist()

    train_indices = shuffled_indices(
        len(train_tensors), torch.Generator().manual_seed(seed),
    )
    auxiliary_indices = shuffled_indices(
        len(auxiliary_dataset), torch.Generator().manual_seed(seed + 1),
    )

    started = time.monotonic()
    with (out / f"{category}_loss.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "loss_st", "loss_ae", "loss_stae", "total", "lr"])
        for step in range(1, steps + 1):
            x = train_tensors[next(train_indices)].cuda()
            aux, _ = auxiliary_dataset[next(auxiliary_indices)]
            aux = aux.unsqueeze(0).cuda()
            lr = optimizer.param_groups[0]["lr"]

            optimizer.zero_grad(set_to_none=True)
            losses = model(batch=x, batch_imagenet=aux)
            total = sum(losses)
            values = torch.stack([*losses, total]).detach().tolist()
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{category}: non-finite loss at step {step}")
            total.backward()
            optimizer.step()
            scheduler.step()
            writer.writerow([step, *values, lr])

            if step == 1 or step % 500 == 0:
                stream.flush()
                print(
                    f"{category}: STEP={step}/{steps} LOSS={values[-1]:.7g} "
                    f"ELAPSED_S={time.monotonic() - started:.1f}",
                    flush=True,
                )

    if any(not torch.isfinite(param).all().item() for param in params):
        raise ValueError(f"{category}: non-finite learned parameters")

    model.eval()
    val_batches = (
        SimpleNamespace(image=tensor, gt_label=torch.zeros(1, dtype=torch.long))
        for tensor in val_tensors
    )
    quantiles = module.map_norm_quantiles(val_batches)
    if (any(not torch.isfinite(v).all().item() for v in quantiles.values())
            or any(quantiles[f"qb_{key}"].item() <= quantiles[f"qa_{key}"].item()
                   for key in ("st", "ae"))):
        raise ValueError(f"{category}: invalid VAL map quantiles")
    model.quantiles.update(quantiles)

    with (out / f"{category}_val_scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "score"])
        with torch.inference_mode():
            for row, tensor in zip(val, val_tensors, strict=True):
                pred = model(tensor.cuda())
                if (pred.pred_score.numel() != 1
                        or tuple(pred.anomaly_map.shape) != (1, 1, 256, 256)
                        or not torch.isfinite(pred.pred_score).all().item()
                        or not torch.isfinite(pred.anomaly_map).all().item()):
                    raise ValueError(f"{category}: invalid VAL output for {row['path']}")
                writer.writerow([row["path"], pred.pred_score.item()])

    checkpoint = out / f"{category}_efficientad_s_fp32.pt"
    partial = checkpoint.with_suffix(".partial")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, partial)
    partial.replace(checkpoint)

    results[category] = {
        "train_images": len(train),
        "val_images": len(val),
        "optimizer_steps": steps,
        "map_quantiles": {key: value.item() for key, value in quantiles.items()},
        "checkpoint": checkpoint.name,
        "elapsed_seconds": time.monotonic() - started,
    }
    print(f"{category}: FP32_TRAIN=PASS / NORMAL_VAL_CALIBRATION=PASS", flush=True)

with (out / "input_sha256.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["category", "role", "path", "sha256"])
    writer.writeheader()
    writer.writerows(used_inputs)

summary = {
    "scope": "FP32 training and normal VAL map calibration",
    "seed": seed,
    "teacher_sha256": smoke_info["teacher_sha256"],
    "imagenette_train_images": len(auxiliary_dataset),
    "imagenette_classes": auxiliary_dataset.classes,
    "batch_size": 1,
    "steps_per_category": steps,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "lr_decay_step": int(0.95 * steps),
    "categories": results,
    "efficientad_test": "NOT_RUN",
}
report = json.dumps(summary, indent=2)
(out / "summary.json").write_text(report + "\n")
print(report)
