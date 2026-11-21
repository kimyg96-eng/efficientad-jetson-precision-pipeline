#!/usr/bin/env bash
set -euo pipefail
mode="${1:-pilot}"
case "$mode" in clockcheck|pilot|full) ;; *) printf 'Usage: bash run.sh clockcheck|pilot|full\n'; exit 2;; esac
package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
audit="$HOME/EdgeAI-Optimization/13_Audit_Evidence"
py="$audit/STEP11C_EFFICIENTAD_CUDA_SMOKE_EAZZ1YA4/venv/bin/python"
(cd "$package" && sha256sum -c --quiet SHA256SUMS.txt)
exec 9>"$audit/.edgeai_pipeline_measurement.lock"
if ! flock -n 9; then printf 'Another pipeline benchmark holds the lock.\n'; exit 1; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset NVIDIA_TF32_OVERRIDE
"$py" -c 'import torch,torchvision,numpy,PIL; print("PYTHON_IMPORTS=PASS")'
trt_site="$(/usr/bin/python3 -c 'import pathlib,tensorrt; print(pathlib.Path(tensorrt.__file__).resolve().parent.parent)')"
out="$(mktemp -d "$audit/STEP23B_${mode^^}_DENSE_ARRIVALS_XXXXXXXX")"
printf 'EVIDENCE_DIR=%s\n' "$out"
cp "$package/"*.py "$package/"*.csv "$package/"*.json "$package/README.txt" "$package/run.sh" "$out/"
restore_needed=0
keepalive=''
finish() {
 code=$?
 trap - EXIT INT TERM
 set +e
 if [ "$restore_needed" -eq 1 ] && [ -f "$out/original_frequency_bounds.json" ]; then
  if sudo -n /usr/bin/python3 "$out/clock_bounds.py" restore "$out" > "$out/clock_bounds_restore.log" 2>&1; then
   printf 'CLOCK_MIN_BOUNDS_RESTORE=PASS_SELECTED_FIELDS_READ_BACK\n'
  else
   printf 'CLOCK_MIN_BOUNDS_RESTORE=FAILED / Inspect clock_bounds_restore.log before further measurement.\n'
   code=1
  fi
 fi
 if [ -n "$keepalive" ]; then kill "$keepalive" 2>/dev/null || true; fi
 /usr/sbin/nvpmodel -q > "$out/power_mode_after.txt"
 (cd "$out" && find . -type f ! -name SHA256SUMS.txt ! -name STEP23B_REVIEW.zip -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt)
 "$py" - "$out" <<'PYZIP'
import sys,zipfile
from pathlib import Path
root=Path(sys.argv[1]);target=root/'STEP23B_REVIEW.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(root.rglob('*')):
  if p.is_file() and p!=target and not p.is_symlink():z.write(p,p.relative_to(root).as_posix())
print(f'UPLOAD_FILE={target}',flush=True)
PYZIP
 archive_code=$?
 if [ "$archive_code" -ne 0 ]; then code=1; fi
 printf 'EXIT=%s / EVIDENCE_DIR=%s\n' "$code" "$out"
 exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
/usr/sbin/nvpmodel -q > "$out/power_mode_before.txt"
"$py" -u "$out/benchmark.py" "$audit" "$out" "$trt_site" --mode preflight 2>&1 | tee "$out/preflight.log"
sudo -v
(exec 9>&-; while true; do sudo -n -v || exit; sleep 45; done) &
keepalive=$!
restore_needed=1
sudo -n /usr/bin/python3 "$out/clock_bounds.py" apply "$out" > "$out/clock_bounds_apply.log" 2>&1
if [ "$mode" = clockcheck ]; then
 cat "$out/clock_bounds_apply.log"
 printf 'CLOCKCHECK_APPLY=PASS / GPU_BENCHMARK=NOT_RUN\n'
 exit 0
fi
"$py" -u "$out/benchmark_open_loop.py" "$audit" "$out" "$trt_site" --mode "$mode" 2>&1 | tee "$out/run.log"

"$py" -u "$out/audit_arrivals.py" "$out" 2>&1 | tee "$out/analysis.log"
