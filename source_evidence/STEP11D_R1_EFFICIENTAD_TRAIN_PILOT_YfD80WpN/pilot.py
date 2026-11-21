import csv
import hashlib
import io
import json
import random
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision import transforms
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

with tarfile.open(assets / "imagenette2.tgz", "r|gz") as archive:
    for member in archive:
        parts = PurePosixPath(member.name).parts
        if (member.isfile() and len(parts) == 4 and parts[1] == "train"
                and PurePosixPath(member.name).suffix.lower() in {".jpeg", ".jpg"}):
            with archive.extractfile(member) as stream:
                auxiliary_bytes = stream.read()
            auxiliary_name = member.name
            break
    else:
        raise SystemExit("No ImageNette training JPEG found")

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

auxiliary = tensor_from_bytes(auxiliary_bytes, auxiliary_transform).cuda()
results = {}
for category, expected in {"bottle": (125, 21), "carpet": (168, 28)}.items():
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

    first_train = read_normal(train[0], category, "TRAIN")

    def teacher_batches():
        yield SimpleNamespace(image=first_train)
        for row in train[1:]:
            yield SimpleNamespace(image=read_normal(row, category, "TRAIN"))

    stats = module.teacher_channel_mean_std(teacher_batches())
    if (any(not torch.isfinite(v).all().item() for v in stats.values())
            or not (stats["std"] > 0).all().item()):
        raise ValueError(f"{category}: invalid teacher statistics")
    model.mean_std.update(stats)

    model.train()
    model.teacher.eval()
    params = [*model.student.parameters(), *model.ae.parameters()]
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5)
    before = params[0].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    losses = model(batch=first_train.cuda(), batch_imagenet=auxiliary)
    if len(losses) != 3 or not all(torch.isfinite(loss).item() for loss in losses):
        raise ValueError(f"{category}: invalid training loss")
    total = sum(losses)
    total.backward()
    optimizer.step()
    if torch.equal(before, params[0]) or any(
        not torch.isfinite(param).all().item() for param in params
    ):
        raise ValueError(f"{category}: invalid optimizer update")

    model.eval()
    x_val = read_normal(val[0], category, "VAL").cuda()
    with torch.inference_mode():
        pred = model(x_val)
    if (pred.pred_score.numel() != 1
            or tuple(pred.anomaly_map.shape) != (1, 1, 256, 256)
            or not torch.isfinite(pred.pred_score).all().item()
            or not torch.isfinite(pred.anomaly_map).all().item()):
        raise ValueError(f"{category}: invalid VAL output")

    results[category] = {
        "teacher_statistics_train_images": len(train),
        "optimizer_steps": 1,
        "loss_st_ae_stae": [loss.item() for loss in losses],
        "val_path": val[0]["path"],
        "val_uncalibrated_score": pred.pred_score.item(),
        "map_shape": list(pred.anomaly_map.shape),
    }
    print(f"{category}: TRAIN_UPDATE=PASS / NORMAL_VAL_FORWARD=PASS", flush=True)

with (out / "input_sha256.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["category", "role", "path", "sha256"])
    writer.writeheader()
    writer.writerows(used_inputs)

summary = {
    "scope": "one-step FP32 training pipeline pilot",
    "seed": seed,
    "teacher_sha256": smoke_info["teacher_sha256"],
    "imagenette_member": auxiliary_name,
    "imagenette_sha256": digest(auxiliary_bytes),
    "categories": results,
    "efficientad_test": "NOT_RUN",
}
report = json.dumps(summary, indent=2)
(out / "summary.json").write_text(report + "\n")
print(report)
