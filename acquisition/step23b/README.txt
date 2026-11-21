STEP23B DENSE SYNTHETIC ARRIVAL GRID

PILOT FIRST:
 bash "$HOME/Downloads/STEP23B_DENSE_ARRIVALS/run.sh" pilot
Bottle and Carpet, new rates 35 and 80 images/s, both engines, one round, 10-second arrival window: 8 runs. Estimate 3–6 minutes with hash preflight, settle/warmup, drain and packaging. Send STEP23B_REVIEW.zip for review before full.

FULL AFTER PILOT REVIEW:
 bash "$HOME/Downloads/STEP23B_DENSE_ARRIVALS/run.sh" full
Four categories, 35/45/55/70/80 images/s, two engines, six rounds, 60-second arrival windows: 240 runs. At least 10-second warmup and 5-second settling; estimate 5–6 hours. The pilot does not launch full automatically.

Same frozen checkpoints, input manifests, scores, engine hashes, four CPU producers, ordered consumer and total capacity five as STEP21B. Same frequency minima control, sensor observer, 150 ms power-gap quality bound and restore logic. No training, engine building, installation, fan changes or offline-core activation. Shared benchmark lock; no concurrent CPU/GPU benchmarks.

Report drops and scheduled-arrival-to-decision curves over 20–220 ms. Legacy period-deadline fields remain for trace continuity only; no application deadline is implied. Explicit budget counts 50/75/100/125/150/175/200 ms use all offered arrivals, including dropped arrivals in the denominator. These budgets were selected before this new acquisition, after reviewing the earlier 15/30/60 experiment; the earlier deadline sweep is post hoc.

Rate order rotates one position each round, and engine order reverses. Six rounds balance engine order but cannot perfectly balance five rate positions; category blocks remain sequential. A tested rate is called zero-drop only if EVERY round has zero capacity and scheduler-late drops. This is a finite observed criterion, not a guaranteed maximum sustainable rate. Do not interpolate an exact critical rate or pool these runs with earlier sessions without labeling session differences. P4/C5 is held fixed; this grid does not measure P6/P8 service capacity.

Power integrates scheduled first arrival through drain/join, no idle subtraction, from three onboard rails. It is not wall-socket energy. At overload, engines may complete different counts and image subsets. Pilot only validates execution; no new rate-grid scientific results exist yet. Camera capture and hard real-time guarantees remain outside scope.

Local validation: preserved open-loop core tests, Python compile and bash syntax. Jetson acquisition not yet tested for this new grid. On completion, audit_arrivals.py checks count conservation/order/replay and reintegrates power, writes deadline_curves.csv and zero_drop_summary.json. Wrapper packages full raw logs to STEP23B_REVIEW.zip and restores selected clock bounds even after normal interruption. Use tmux for SSH robustness.
