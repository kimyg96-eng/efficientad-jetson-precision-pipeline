import hashlib
import importlib.metadata as metadata
import io
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

import torch
import torchvision
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel

archive, out, base_site, pilot_site = (Path(value) for value in sys.argv[1:])

if (
    torch.__version__ != "2.12.0+cu132"
    or torchvision.__version__ != "0.27.0+cu132"
    or metadata.version("anomalib") != "2.3.0"
    or not Path(torch.__file__).resolve().is_relative_to(base_site.resolve())
    or not Path(torchvision.__file__).resolve().is_relative_to(pilot_site.resolve())
):
    raise SystemExit("Unexpected pinned package version or import location")

with zipfile.ZipFile(archive) as bundle:
    members = [
        entry for entry in bundle.infolist()
        if not entry.is_dir()
        and entry.filename == 'efficientad_pretrained_weights/pretrained_teacher_small.pth'
    ]
    if len(members) != 1:
        raise SystemExit("Expected exactly one EfficientAD-S teacher in the verified ZIP")
    teacher_bytes = bundle.read(members[0])

torch.manual_seed(20261116)
model = EfficientAdModel().eval()
state = torch.load(io.BytesIO(teacher_bytes), map_location="cpu", weights_only=True)
model.teacher.load_state_dict(state, strict=True)
model.cuda()

x = torch.rand((1, 3, 256, 256), device="cuda", dtype=torch.float32)
with torch.inference_mode():
    result = model(x)
torch.cuda.synchronize()

outputs = {}
for key in ("pred_score", "anomaly_map"):
    value = getattr(result, key)
    if not isinstance(value, torch.Tensor) or not value.is_cuda:
        raise SystemExit(f"Expected a CUDA tensor for {key}")
    if not torch.isfinite(value).all().item():
        raise SystemExit(f"Non-finite values in {key}")
    outputs[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}

if result.pred_score.numel() != 1 or tuple(result.anomaly_map.shape) != (1, 1, 256, 256):
    raise SystemExit(f"Unexpected output shapes: {outputs}")

summary = {
    "anomalib": metadata.version("anomalib"),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "teacher_zip_member": members[0].filename,
    "teacher_sha256": hashlib.sha256(teacher_bytes).hexdigest(),
    "teacher_strict_load": "PASS",
    "synthetic_cuda_forward": "PASS",
    "outputs": outputs,
    "training": "NOT_RUN",
    "efficientad_test": "NOT_RUN",
}
report = json.dumps(summary, indent=2)
(out / "summary.json").write_text(report + "\n")
print(report)
