# STEP23C Dense arrival full experiment review

240 prespecified arms completed in a fresh full session; the interrupted 35-arm session and pilot are excluded. Six rounds per condition, 60 seconds offered arrivals, P4 and total capacity five. Do not pool condition means with earlier 15/30/60 sessions.

|Category|TF32 drop at 35|FP16 largest tested zero-drop rate|FP16 drop at 70|FP16 drop at 80|
|---|---:|---:|---:|---:|
|bottle|3.286%|80|0.000%|0.000%|
|carpet|3.587%|55|13.992%|18.708%|
|capsule|3.556%|55|0.869%|17.813%|
|grid|3.246%|80|0.000%|0.000%|

Largest tested zero-drop rate requires zero drops in every round. Bottle/Grid 80 is an observed lower bound on attainable zero-drop rate, not a measured upper limit. Carpet/Capsule show no drops through 55 and drops at 70; the onset is not exactly identified. The earlier 60-rate zero-drop finding is from a separate session. TF32 dropped at every rate in this new grid; the earlier 30-rate result provides separate contextual evidence, not a pooled boundary estimate.

Deadline curves count all offered inputs including drops. Summary ranges are observed run minima/maxima, not confidence intervals. Preprocessing, schedule, finite capacity, ordered consumption and clock policy jointly determine delivery. No camera or hard-real-time guarantee.

Audit: 1476 hashes; 639532 decisions; sensor errors 0; maximum gap 56.751645 ms; energy replay error 0 J. Clock restoration passed. Raw archive: STEP23B_REVIEW(1).zip. Reproduce: python analyze.py <extracted_full_review> <output>; trusted power_monitor.py is included in this analysis folder for portable use.
