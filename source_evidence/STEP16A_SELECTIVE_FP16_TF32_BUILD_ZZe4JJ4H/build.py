import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

audit, out = map(Path, sys.argv[1:])
trtexec = Path("/usr/bin/trtexec")
if not trtexec.is_file():
    raise FileNotFoundError(trtexec)

sources = {
    "all_conv_fp16":
        audit / "STEP11L_EFFICIENTAD_CONV_FP16_2aQndiJ2",
    "teacher_conv_fp16":
        audit / "STEP11M_EFFICIENTAD_BRANCH_FP16_oetzGryV/teacher",
    "student_conv_fp16":
        audit / "STEP11M_EFFICIENTAD_BRANCH_FP16_oetzGryV/student",
    "autoencoder_conv_fp16":
        audit / "STEP11M_EFFICIENTAD_BRANCH_FP16_oetzGryV/autoencoder",
    "ae_encoder_conv_fp16":
        audit / "STEP11N_EFFICIENTAD_AE_SPLIT_FP16_B1HhoFFr/encoder",
    "ae_decoder_conv_fp16":
        audit / "STEP11N_EFFICIENTAD_AE_SPLIT_FP16_B1HhoFFr/decoder",
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

jobs = []
for category in ("bottle", "carpet"):
    for configuration, folder in sources.items():
        source = folder / f"{category}_efficientad_s_conv_fp16.onnx"
        if not source.is_file():
            raise FileNotFoundError(source)
        jobs.append({
            "category": category,
            "configuration": configuration,
            "source_onnx": str(source),
            "source_sha256": digest(source),
        })

env = os.environ.copy()
previous_override = env.pop("NVIDIA_TF32_OVERRIDE", None)
protocol = {
    "change": "Remove --noTF32 from strongly typed mixed ONNX builds",
    "tf32_policy": "Allowed; actual tactics require inspection",
    "original_environment_tf32_override": previous_override,
    "build_environment_tf32_override": None,
    "jobs": jobs,
}
(out / "protocol.json").write_text(
    json.dumps(protocol, indent=2) + "\n"
)

completed = []
for job in jobs:
    category = job["category"]
    configuration = job["configuration"]
    source = Path(job["source_onnx"])
    directory = out / category / configuration
    directory.mkdir(parents=True)
    engine = directory / "model.engine"

    command = [
        str(trtexec),
        f"--onnx={source}",
        f"--saveEngine={engine}",
        "--stronglyTyped",
        "--skipInference",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={directory / 'layer_info.json'}",
    ]
    (directory / "build.command.txt").write_text(
        shlex.join(command) + "\n"
    )
    print(f"BUILD_START={category}/{configuration}", flush=True)
    with (directory / "build.log").open("w") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT,
            env=env, check=True,
        )

    if not engine.is_file() or engine.stat().st_size == 0:
        raise RuntimeError(f"Missing engine: {engine}")
    layer_info = directory / "layer_info.json"
    json.loads(layer_info.read_text())
    if digest(source) != job["source_sha256"]:
        raise RuntimeError(f"Source changed during build: {source}")

    completed.append({
        **job,
        "engine": str(engine.relative_to(out)),
        "engine_sha256": digest(engine),
    })
    (out / "summary.json").write_text(
        json.dumps({"completed": completed}, indent=2) + "\n"
    )
    print(f"BUILD_COMPLETE={category}/{configuration}", flush=True)

print(f"ENGINES_BUILT={len(completed)}/12", flush=True)
