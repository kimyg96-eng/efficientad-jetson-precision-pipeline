# STEP23A Exploratory deadline and queueing analysis

Run: python3 deadline_analysis.py /path/to/step21b_full_review --out analysis
Pass --capacity-source /path/to/extracted_STEP22A; otherwise use source.parent/step22a_complete. The sample-path table is required by this script. This script reads existing logs; it does not create new arrival-rate observations.

Deadline sweep 20–220 ms, step 5 ms, was added after reviewing fixed-rate results and is explicitly post hoc. Curves use all offered arrivals as denominator; dropped arrivals never contribute on-time decisions. Completed-only CDF is exported separately and must not be confused with delivery fraction. Means and shaded minima/maxima use six runs, not independent corpus images. A 100 ms sensitivity case existed before this post hoc sweep; it is not an application-derived deadline. No hard-real-time guarantee.

little_sample_path.csv computes finite-window read-start-to-decision occupancy from the sum of observed image sojourns, including completed drain. It checks an accounting identity Lbar = X Wbar. This is not an independent validation of queueing theory, does not set Lbar equal to configured capacity, and predicts no p95. Throughput uses the same finite host window as its occupancy integral. Waiting before read start is excluded, matching closed-loop latency.
