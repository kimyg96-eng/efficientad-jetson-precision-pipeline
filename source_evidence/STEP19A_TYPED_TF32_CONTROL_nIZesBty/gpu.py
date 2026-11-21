import csv
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

audit, build, out = map(Path, sys.argv[1:])
shared = audit / "STEP13E_EFFICIENTAD_MATCHED_GPU_LATENCY_KtNxqU58/shared_input.bin"
if not shared.is_file() or shared.stat().st_size != 1 * 3 * 256 * 256 * 4:
    raise ValueError("Unexpected shared FP32 input")

jobs = json.loads((build / "summary.json").read_text())["completed"]
env = os.environ.copy()
env.pop("NVIDIA_TF32_OVERRIDE", None)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

protocol = {
    "rounds": 4,
    "warmup_ms": 1000,
    "minimum_duration_s": 5,
    "minimum_iterations": 50,
    "scope": "GPU compute, transfers excluded",
    "input_sha256": sha(shared),
    "engines": [],
}
rows = []
tactics = []

for category in ("bottle", "carpet"):
    engines = {
        "strong_tf32":
            audit / "STEP13A_R1_EFFICIENTAD_TF32_REFERENCES_G2Jd0c1J"
            / category / "strongly_typed_tf32_allowed/model.engine",
    }
    for job in jobs:
        if job["category"] != category:
            continue
        path = build / job["engine"]
        if sha(path) != job["engine_sha256"]:
            raise ValueError(f"Engine hash mismatch: {path}")
        engines[job["configuration"]] = path

        info = json.loads((path.parent / "layer_info.json").read_text())
        layers = info.get("Layers", []) if isinstance(info, dict) else info
        for layer in layers:
            if isinstance(layer, dict):
                tactics.append({
                    "category": category,
                    "configuration": job["configuration"],
                    "layer": layer,
                })

    if len(engines) != 7:
        raise ValueError(f"Expected seven engines: {category}")
    for name, path in engines.items():
        protocol["engines"].append({
            "category": category, "configuration": name,
            "path": str(path), "sha256": sha(path),
        })
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )

    names = list(engines)
    for round_index in range(4):
        order = names[round_index:] + names[:round_index]
        if round_index % 2:
            order.reverse()
        for name in order:
            directory = out / category / name
            directory.mkdir(parents=True, exist_ok=True)
            prefix = directory / f"round_{round_index + 1}"
            times = prefix.with_suffix(".times.json")
            command = [
                "/usr/bin/trtexec",
                f"--loadEngine={engines[name]}",
                f"--loadInputs=image:{shared}",
                "--noDataTransfers", "--infStreams=1",
                "--warmUp=1000", "--duration=5", "--iterations=50",
                f"--exportTimes={times}",
            ]
            prefix.with_suffix(".command.txt").write_text(
                shlex.join(command) + "\n"
            )
            with prefix.with_suffix(".log").open("w") as stream:
                subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT,
                    env=env, check=True,
                )
            records = json.loads(times.read_text())
            values = [float(record["computeMs"]) for record in records]
            if not values or any(
                not (0 < value < float("inf")) for value in values
            ):
                raise ValueError(f"Invalid GPU timings: {times}")
            median = statistics.median(values)
            rows.append([category, name, round_index + 1, len(values), median])
            print(
                f"RUN_COMPLETE={category}/{name}/{round_index + 1}"
                f" GPU_MEDIAN_MS={median:.6f}", flush=True
            )
            with (out / "run_medians.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "category", "configuration", "round",
                    "timed_samples", "gpu_median_ms",
                ])
                writer.writerows(rows)

with (out / "summary.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow([
        "category", "configuration", "median_of_run_medians_ms",
        "min_run_median_ms", "max_run_median_ms",
        "mean_paired_reduction_vs_strong_tf32_pct",
    ])
    for category in ("bottle", "carpet"):
        baseline = {
            row[2]: row[4] for row in rows
            if row[0] == category and row[1] == "strong_tf32"
        }
        for name in dict.fromkeys(r[1] for r in rows if r[0] == category):
            selected = [r for r in rows if r[0] == category and r[1] == name]
            values = [r[4] for r in selected]
            reduction = statistics.mean(
                100 * (baseline[r[2]] - r[4]) / baseline[r[2]]
                for r in selected
            )
            writer.writerow([
                category, name, statistics.median(values),
                min(values), max(values), reduction,
            ])
            print(
                f"{category}/{name}: GPU_MS={statistics.median(values):.6f}"
                f" REDUCTION_VS_STRONG_TF32_PCT={reduction:+.4f}", flush=True
            )

(out / "layer_inventory.json").write_text(
    json.dumps(tactics, indent=2) + "\n"
)
print(f"RUNS_COMPLETE={len(rows)}/56", flush=True)
