import csv
import json
import shlex
import sys
from pathlib import Path

pixel, timing, build, manifest, out, helper_dir, *verified_engine_dirs = (
    Path(p).resolve(strict=True) for p in sys.argv[1:]
)
categories = ("bottle", "carpet", "capsule", "grid")
policies = ("strict_fp32", "default_tf32", "automatic_fp16")
engines = {}

for category in categories:
    if category in {"bottle", "carpet"}:
        paths = {}
        for policy, old_name, verified_dir in zip(
            policies, ("original_no_tf32", "default_tf32", "automatic_fp16"),
            verified_engine_dirs, strict=True,
        ):
            command = timing / category / f"round_1_{old_name}.command.txt"
            tokens = shlex.split(command.read_text())
            flags = [a.partition("=")[2] for a in tokens if a.startswith("--loadEngine=")]
            if len(flags) != 1:
                raise ValueError(f"Expected one engine path in {command}")
            path = Path(flags[0]).resolve(strict=True)
            if not path.is_relative_to(verified_dir):
                raise ValueError(f"Engine outside its verified source directory: {path}")
            paths[policy] = str(path)
        engines[category] = paths
    else:
        engines[category] = {
            policy: str((build / category / policy / "model.engine").resolve(strict=True))
            for policy in policies
        }

(out / "engine_paths.json").write_text(json.dumps(engines, indent=2) + "\n")

with manifest.open(newline="") as stream:
    rows = [r for r in csv.DictReader(stream) if r["category"] in categories]
fields = ("category", "image_path", "image_sha256", "defect",
          "anomalous", "mask_path", "mask_sha256")
with (out / "test_manifest.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        category = row["category"]
        parts = Path(row["path"]).parts
        if (len(parts) != 4 or parts[:2] != (category, "test")
                or row["label"] not in {"0", "1"}
                or (parts[2] == "good") != (row["label"] == "0")):
            raise ValueError(f"Unexpected test path or label: {row['path']}")
        writer.writerow(dict(
            category=category, image_path=row["path"], image_sha256=row["sha256"],
            defect=parts[2], anomalous=row["label"], mask_path=row["mask_path"],
            mask_sha256=row["mask_sha256"],
        ))

source = (pixel / "evaluate.py").read_text()

def replace(old, new):
    global source
    if source.count(old) != 1:
        raise ValueError(f"Unexpected STEP11R source pattern: {old!r}")
    source = source.replace(old, new, 1)

replace(
    'fp32, split, data, out, trt_site = (Path(arg).resolve() for arg in sys.argv[1:])',
    'pixel, split, data, out, trt_site, old_test, new_test = (\n'
    '    Path(arg).resolve(strict=True) for arg in sys.argv[1:]\n'
    ')\n'
    'engine_paths = json.loads((out / "engine_paths.json").read_text())\n'
    'historical = json.loads((pixel / "summary.json").read_text())["categories"]\n'
    'controls = {}\n'
    'for folder in (old_test, new_test):\n'
    '    with (folder / "test_scores.csv").open(newline="") as stream:\n'
    '        for row in csv.DictReader(stream):\n'
    '            key = (row["category"], row["path"])\n'
    '            if key in controls:\n'
    '                raise ValueError(f"Duplicate score control: {key}")\n'
    '            controls[key] = row\n'
    'if len(controls) != 410:\n'
    '    raise ValueError("Unexpected score-control inventory")\n',
)
replace(
    'import tensorrt as trt\n',
    'import tensorrt as trt\n'
    'if trt.__version__ != "10.16.2.10":\n'
    '    raise RuntimeError(f"Unexpected TensorRT version: {trt.__version__}")\n',
)
replace('source = split / "verify_conv_fp16.py"', 'source = split / "verify_fp16.py"')
replace('Expected one STEP11N open_engine function', 'Expected one STEP11J open_engine function')
replace(
    '{"bottle": (20, 63), "carpet": (28, 89)}.items()',
    '{"bottle": (20, 63), "carpet": (28, 89), '
    '"capsule": (23, 109), "grid": (21, 57)}.items()',
)
replace(
    '    engines = {"fp32": fp32 / f"{category}_efficientad_s_fp32.engine"}\n'
    '    if category == "carpet":\n'
    '        engines["decoder_conv_fp16"] = (\n'
    '            split / "decoder/carpet_efficientad_s_conv_fp16.engine"\n'
    '        )',
    '    engines = {name: Path(path) for name, path in engine_paths[category].items()}',
)
replace(
    '        prefix = out / f"{category}_{name}"',
    '        column = ("original_no_tf32"\n'
    '                  if name == "strict_fp32" and category in {"bottle", "carpet"}\n'
    '                  else name)\n'
    '        matched = [controls[category, row["image_path"]] for row in rows]\n'
    '        if any(r["label"] != t["anomalous"] for r, t in zip(matched, rows, strict=True)):\n'
    '            raise ValueError(f"Score-control label mismatch: {category}")\n'
    '        expected_scores = np.array([float(r[column]) for r in matched])\n'
    '        max_delta = float(np.max(np.abs(scores - expected_scores)))\n'
    '        if not np.isfinite(max_delta) or max_delta > 1e-6:\n'
    '            raise ValueError(f"Score replay mismatch: {category}/{name}: {max_delta}")\n'
    '        print(f"SCORE_REPLAY={category}/{name} MAX_ABS={max_delta:.9g}", flush=True)\n'
    '        prefix = out / f"{category}_{name}"',
)
replace(
    '        results[category]["engines"][name] = result',
    '        if category in historical and name == "strict_fp32":\n'
    '            prior = historical[category]["engines"]["fp32"]["pixel"]\n'
    '            delta = max(abs(result["pixel"][m] - prior[m]) for m in ("auroc", "ap"))\n'
    '            if delta > 1e-8:\n'
    '                raise ValueError(f"STEP11R pixel metric replay mismatch: {category}: {delta}")\n'
    '            print(f"STEP11R_PIXEL_REPLAY={category} MAX_ABS={delta:.9g}", flush=True)\n'
    '        results[category]["engines"][name] = result',
)
source = source[:source.index('\ncarpet = results["carpet"]["engines"]')] + '''
report = {
    "categories": results,
    "pixel_protocol": "native-resolution bilinear interpolation, align_corners=False; pooled by category",
    "versions": {"tensorrt": trt.__version__, "torch": torch.__version__,
                 "scikit_learn": sklearn.__version__},
}
(out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\\n")
with (out / "pixel_metrics.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["category", "engine", "images", "pixels", "positive_pixels",
                     "pixel_auroc", "pixel_ap"])
    for category, entry in results.items():
        for name, metrics_by_level in entry["engines"].items():
            pixel_result = metrics_by_level["pixel"]
            writer.writerow([category, name, entry["test_images"], entry["pixels"],
                             entry["positive_pixels"], pixel_result["auroc"], pixel_result["ap"]])
'''
target = out / "evaluate_pixel.py"
compile(source, str(target), "exec")
target.write_text(source)
