import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

out = Path(sys.argv[1])
wheels = list((out / "wheels").glob("anomalib-2.3.0-*.whl"))
if len(wheels) != 1:
    raise SystemExit("Expected exactly one Anomalib 2.3.0 wheel")

source_path = "anomalib/models/image/efficient_ad/lightning_model.py"
with zipfile.ZipFile(wheels[0]) as wheel:
    source = wheel.read(source_path)
(out / "upstream_efficientad_lightning_model.py").write_bytes(source)

expected_assets = {
    "WEIGHTS_DOWNLOAD_INFO": "efficientad_pretrained_weights.zip",
    "IMAGENETTE_DOWNLOAD_INFO": "imagenette2.tgz",
}
assets = {}
for node in ast.parse(source).body:
    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
        continue
    for target in node.targets:
        if isinstance(target, ast.Name) and target.id in expected_assets:
            assets[target.id] = {
                kw.arg: ast.literal_eval(kw.value)
                for kw in node.value.keywords
                if kw.arg in {"name", "url", "hashsum"}
            }

if assets.keys() != expected_assets.keys():
    raise SystemExit("Could not locate both upstream asset declarations")

for key, name in expected_assets.items():
    info = assets[key]
    url, expected = info["url"], info["hashsum"]
    if (info["name"] != name or urlsplit(url).scheme != "https"
            or not re.fullmatch(r"[0-9a-f]{64}", expected)):
        raise SystemExit(f"Unexpected upstream declaration: {key}")

    partial = out / "assets" / f"{name}.partial"
    subprocess.run(
        ["curl", "--fail", "--location", "--retry", "2",
         "--connect-timeout", "15", "--proto", "=https",
         "--proto-redir", "=https", "--output", str(partial), url],
        check=True,
    )
    with partial.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch for {name}: {actual}")
    partial.rename(out / "assets" / name)
    print(f"{name}: UPSTREAM_SHA256=PASS", flush=True)

summary = {
    "implementation": "anomalib==2.3.0",
    "model_size": "S",
    "pilot_categories": ["bottle", "carpet"],
    "upstream_source": source_path,
    "assets": assets,
    "training": "NOT_RUN",
    "efficientad_test": "NOT_RUN",
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
