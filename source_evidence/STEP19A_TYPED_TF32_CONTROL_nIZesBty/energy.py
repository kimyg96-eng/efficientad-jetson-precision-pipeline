import ast
import ctypes as C
import subprocess
import csv
import hashlib
import io
import json
import re
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

train, fp32, split, data, out, trt_site, candidate_path = map(Path, sys.argv[1:8])
category, configuration = sys.argv[8:10]
expected_val = {"bottle": 21, "carpet": 28}[category]
sys.path.append(str(trt_site))
import tensorrt as trt

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
input_shape = (1, 3, 256, 256)
output_names = ("pred_score", "anomaly_map")
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

source = split / "verify_conv_fp16.py"
tree = ast.parse(source.read_text())
required = {"load_input", "open_engine"}
functions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in required
]
if len(functions) != 2 or {node.name for node in functions} != required:
    raise ValueError("Expected STEP11N input and engine helpers")
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"))

with (train / "input_sha256.csv").open(newline="") as file:
    rows = [
        row for row in csv.DictReader(file)
        if row["category"] == category and row["role"] == "VAL"
    ]
if len(rows) != expected_val or len({row["path"] for row in rows}) != expected_val:
    raise ValueError(f"Unexpected {category} VAL inventory")
images = [np.ascontiguousarray(load_input(row, category).numpy()) for row in rows]
cuda = C.CDLL("libcudart.so.13")
copy_async = cuda.cudaMemcpyAsync
copy_async.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p]
copy_async.restype = C.c_int

def copy(dst, src, size, direction, stream):
    error = copy_async(dst, src, size, direction, stream.cuda_stream)
    if error:
        raise RuntimeError(f"cudaMemcpyAsync failed with CUDA error {error}")

def check_mode():
    mode = subprocess.check_output(["nvpmodel", "-q"], text=True).strip()
    if "NV Power Mode: MODE_30W" not in mode.splitlines():
        raise RuntimeError(f"Expected MODE_30W. Observed {mode!r}")
    return mode

gpu_clock = Path("/sys/class/devfreq/17000000.gpu/cur_freq")
emc_clock = Path("/sys/class/devfreq/bwmgr/cur_freq")
for clock in (gpu_clock, emc_clock):
    if int(clock.read_text()) <= 0:
        raise ValueError(f"Invalid clock reading from {clock}")

thermal = [
    (zone.name, (zone / "type").read_text().strip(), zone / "temp")
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*"))
    if (zone / "type").is_file() and (zone / "temp").is_file()
]
thermal = [row for row in thermal if any(k in row[1].lower() for k in ("cpu", "gpu", "tj"))]
if not thermal:
    raise RuntimeError("No CPU, GPU, or junction thermal sensor found")

(out / "protocol.json").write_text(json.dumps({
    "power_mode": check_mode(),
    "category": category, "configuration": configuration,
    "candidate_engine": str(candidate_path),
    "reference_engine": str(fp32 / f"{category}_efficientad_s_fp32.engine"),
    "reference": "strongly typed TF32, TF32 permitted",
    "warmup_seconds": 10, "minimum_measurement_seconds": 60,
    "settle_seconds": 5, "pairs": 8, "batch": 1,
    "order": ["strong_tf32->candidate", "candidate->strong_tf32"],
    "val_order": [row["path"] for row in rows],
    "gpu_scope": "CUDA events around TensorRT enqueue on the same stream",
    "host_scope": "pageable CPU input, H2D, inference, both D2H outputs, synchronize",
    "clock_policy": "observed only; no clock settings changed",
    "thermal_sensors": {key: label for key, label, _ in thermal},
}, indent=2) + "\n")

rails = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")
sensors = {}
for label in sorted(Path("/sys/class/hwmon").glob("hwmon*/in*_label")):
    rail = label.read_text().strip()
    match = re.fullmatch(r"in(\d+)_label", label.name)
    if rail not in rails or match is None:
        continue
    if rail in sensors:
        raise ValueError(f"Duplicate sensor for {rail}")
    index = match[1]
    voltage = label.with_name(f"in{index}_input")
    current = label.with_name(f"curr{index}_input")
    sensors[rail] = (voltage.resolve(), current.resolve())
if set(sensors) != set(rails):
    raise ValueError(f"Missing expected rails. Found: {list(sensors)}")
(out / "sensors.json").write_text(json.dumps(
    {rail: [str(path) for path in sensors[rail]] for rail in rails}, indent=2
) + "\n")

def sample():
    before = time.monotonic_ns()
    values = [
        int(path.read_text())
        for rail in rails
        for path in sensors[rail]
    ]
    after = time.monotonic_ns()
    return ((before + after) // 2, *values)

for rail, voltage in zip(rails, sample()[1::2], strict=True):
    if voltage <= 0:
        raise ValueError(f"Invalid idle voltage for {rail}")

engines = {
    "strong_tf32": fp32 / f"{category}_efficientad_s_fp32.engine",
    "candidate": candidate_path,
}
orders = [("strong_tf32", "candidate"), ("candidate", "strong_tf32")] * 4
results = []
for pair, order in enumerate(orders, start=1):
    for name in order:
        engine, context, buffers = open_engine(engines[name], category)
        stream = torch.cuda.Stream()

        check_mode()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        host_outputs = {
            key: np.empty(tuple(buffers[key].shape), dtype=np.float32)
            for key in output_names
        }

        def infer(image, timed=False):
            t0 = time.monotonic_ns()
            copy(buffers["image"].data_ptr(), image.ctypes.data, image.nbytes, 1, stream)
            if timed:
                start_event.record(stream)
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError(f"TensorRT execution failed for {name}")
            if timed:
                end_event.record(stream)
            for key, output in host_outputs.items():
                copy(output.ctypes.data, buffers[key].data_ptr(), output.nbytes, 2, stream)
            stream.synchronize()
            t1 = time.monotonic_ns()
            if timed:
                return t0, t1, start_event.elapsed_time(end_event), t1 - t0

        samples, clocks, temperatures = [], [], []
        timings = []
        errors = []
        stop = threading.Event()

        def collect():
            try:
                deadline = time.monotonic_ns()
                last_temperature = 0
                while not stop.is_set():
                    if stop.wait(max(0, deadline - time.monotonic_ns()) / 1e9):
                        break
                    samples.append(sample())
                    c0 = time.monotonic_ns()
                    gpu_hz, emc_hz = (int(path.read_text()) for path in (gpu_clock, emc_clock))
                    c1 = time.monotonic_ns()
                    clocks.append(((c0 + c1) // 2, gpu_hz, emc_hz))
                    if c1 - last_temperature >= 1_000_000_000:
                        temperatures.append((c1, *(int(path.read_text()) for _, _, path in thermal)))
                        last_temperature = c1
                    deadline = max(deadline + 50_000_000, time.monotonic_ns() + 1)
            except Exception as exc:
                errors.append(exc)

        with torch.cuda.stream(stream):
            time.sleep(5)
            warm_end = time.monotonic() + 10
            warm_index = 0
            while time.monotonic() < warm_end:
                infer(images[warm_index % len(images)])
                warm_index += 1

            samples.append(sample())
            worker = threading.Thread(target=collect)
            worker.start()
            try:
                start = time.monotonic_ns()
                deadline = start + 60_000_000_000
                while not timings or timings[-1][2] < deadline:
                    index = len(timings) % len(images)
                    timings.append((index, *infer(images[index], timed=True)))
                end = timings[-1][2]
            finally:
                stop.set()
                worker.join()
                samples.append(sample())

        prefix = out / f"pair{pair}_{name}"
        with prefix.with_suffix(".power.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["monotonic_ns", *(
                f"{rail}_{unit}" for rail in rails for unit in ("mV", "mA")
            )])
            writer.writerows(samples)
        with prefix.with_suffix(".latency.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["val_index", "start_ns", "end_ns", "gpu_ms", "host_ns"])
            writer.writerows(timings)
        with prefix.with_suffix(".clocks.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["monotonic_ns", "gpu_hz", "emc_hz"])
            writer.writerows(clocks)
        with prefix.with_suffix(".temperature.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["monotonic_ns", *(key + "_mC" for key, _, _ in thermal)])
            writer.writerows(temperatures)
        check_mode()
        if errors:
            raise RuntimeError("Power sampling failed") from errors[0]

        t = np.array([(s[0] - start) / 1e9 for s in samples])
        duration = (end - start) / 1e9
        if not t[0] < 0 < duration < t[-1] or not np.all(np.diff(t) > 0):
            raise ValueError("Power samples do not bracket the workload")
        measured_gaps = np.diff(t)
        within_window = (t[:-1] < duration) & (t[1:] > 0)
        max_gap = float(measured_gaps[within_window].max())
        if max_gap > 0.15:
            raise ValueError(f"Power sampling gap too large: {max_gap:.3f} s")

        boundaries = np.r_[0.0, t[(t > 0) & (t < duration)], duration]
        readings = np.array([s[1:] for s in samples], dtype=np.float64)
        power = readings[:, 0::2] * readings[:, 1::2] / 1e6
        energy = {
            rail: float(np.trapezoid(np.interp(boundaries, t, power[:, i]), boundaries))
            for i, rail in enumerate(rails)
        }
        result = {
            "pair": pair,
            "engine": name,
            "start_ns": start,
            "end_ns": end,
            "duration_s": duration,
            "inferences": len(timings),
            "median_latency_ms": statistics.median(row[4] for row in timings) / 1e6,
            "median_gpu_ms": statistics.median(row[3] for row in timings),
            "max_power_gap_ms": max_gap * 1000,
            "energy_j_by_rail": energy,
            "rail_sum_mj_per_inference": sum(energy.values()) * 1000 / len(timings),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        del context, engine, buffers

pairs = []
for pair in range(1, 9):
    items = {row["engine"]: row for row in results if row["pair"] == pair}
    ref, candidate = items["strong_tf32"], items["candidate"]
    pairs.append({
        "pair": pair,
        "gpu_reduction_pct": 100 * (
            1 - candidate["median_gpu_ms"] / ref["median_gpu_ms"]
        ),
        "latency_reduction_pct": 100 * (
            1 - candidate["median_latency_ms"] / ref["median_latency_ms"]
        ),
        "energy_reduction_pct": 100 * (
            1 - candidate["rail_sum_mj_per_inference"] /
            ref["rail_sum_mj_per_inference"]
        ),
    })
report = {
    "category": category,
    "configuration": configuration,
    "val_images": len(images),
    "measurement": "GPU event latency and host latency including H2D, two D2H outputs and synchronization",
    "power_scope": "sum of three on-board rails; no idle subtraction",
    "runs": results,
    "paired_results": pairs,
    "efficientad_test": "NOT_RUN",
}
(out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(pairs, indent=2))
for key in ("gpu_reduction_pct", "latency_reduction_pct", "energy_reduction_pct"):
    values = [row[key] for row in pairs]
    print(f"{key}: mean={statistics.mean(values):.6f} min={min(values):.6f} max={max(values):.6f}")
