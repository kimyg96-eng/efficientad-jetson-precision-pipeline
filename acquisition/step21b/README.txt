STEP21B fixed-arrival pipeline pilot

Run: bash run.sh clockcheck ; then bash run.sh pilot
Only run full after the pilot review. Do not run other GPU work concurrently.
Pilot: Bottle and Carpet, 15 and 60 arrivals/s, 2 engines, 1 round, 10 s offered arrivals = 8 arms.
Full: 4 categories, 15/30/60 arrivals/s, 2 engines, 6 rounds, 60 s arrivals = 144 arms.
P4 preprocessing, one TensorRT consumer, total in-flight capacity 5 for every arm.
Primary deadline is one arrival period; 100 ms is a prespecified sensitivity deadline. These are synthetic load/deadline scenarios, not application requirements.
The arrival thread uses absolute scheduled times and never waits for capacity. New arrivals are dropped if full. If release dispatch is at least one period late, that arrival is dropped separately; no catch-up bursts. Admission and capacity drops are fully enumerated.
Latency starts at the scheduled arrival, not preprocessing admission. Report on-time completion / all offered arrivals, with drop rate and completed-image deadline misses separately. Accepted images drain after arrivals stop. No image is silently discarded.
All scheduled arrivals and completed images are logged. Three-rail energy includes all scheduled-arrival time and drain, without idle subtraction. mJ/offered, mJ/completed and mJ/on-time have distinct denominators. Undefined denominators produce null.
Clock bounds use the corrected STEP20A apply/wait/restore sequence; no governors, CPU idle states, fan controls, or power mode are changed. Saved scores and validation thresholds remain frozen. No retraining, engine builds or package installations.
The provided CPU tests exercise scheduling, overload drops and worker failure cleanup. Jetson/TensorRT execution is untested until the pilot. Exceptions invalidate the arm and create a review ZIP; raw partial records are retained where available.
The preflight benchmark.py uses the previously verified frozen image/engine hashes and exits before inference. benchmark_open_loop.py performs acquisition. The original closed-loop core is used only for untimed warmup.
This package does not implement P6/P8 or capacity sweeps; those require separately controlled experiments.
