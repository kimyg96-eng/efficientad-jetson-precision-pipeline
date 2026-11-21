# STEP23D Drop categories and completed-image stage analysis

Analysis of the accepted STEP23B full session only: 240 arms and six rounds per condition. No new GPU measurements. Source: STEP23B_REVIEW(1).zip. Run python analysis/step23d/analyze.py /path/to/extracted_STEP23B_FULL --out /path/to/output. Requires numpy.

All offered = completed + capacity drops + scheduler-late drops. For completed images the scheduled-to-decision duration decomposes into release lateness + release-to-read + read/preparation + ready wait + consumer service. This identity is verified per image within 1e-6 ms. Means are averages of run means; min/max are observed across runs. Do not add component medians or p95 values.

Consumer busy fraction uses measured service duration divided by the finite host window through drain/join. It is not a CUDA utilization counter. Dropped images have no preparation/service trace, so completed-image component means cannot establish their counterfactual costs. Zero scheduler-late drops excludes that recorded rejection category; it does not eliminate all CPU scheduling effects on processing. Capacity rejection identifies the admission rule, not a unique hardware root cause. Comparing open-loop service with earlier closed-loop capacity does not isolate finite-capacity, arrival phasing, ordered consumption or preparation variability.
