STEP22A R1 RECOVERY — INTERRUPTED FULL PRODUCER/CAPACITY EXPERIMENT

WHY
The reviewed run completed 493/512 arms and then failed on grid automatic_fp16 P8/C10/R7 because a 222.316761 ms power-sample gap exceeded the existing 150 ms quality limit. Frequency-bound restoration passed. This package does NOT relax the gap threshold, interpolate across an accepted larger limit, or change power sampling, producer code, engine code, preprocessing, or acquisition durations.

RECOVERY PLAN
Retain 492 existing valid arms. Reacquire the incomplete engine pair (both grid/default_tf32 and grid/automatic_fp16 P8/C10/R7), then the remaining P4/C1 pair in round 7 and all 16 grid round-8 arms. Total new arms: 20. The original successful TF32 P8/C10/R7 arm is superseded ONLY in the new combined analysis so its pair is not split across sessions. Its original raw files, and failed-arm raw files, remain untouched in the original source directory and original review ZIP.
The plan is selected only by completeness/quality, never by whether FP16 improves a metric. Paired reacquisition is declared before viewing the new results. No automatic retries or early stopping based on performance.

RUN
After downloading and extracting STEP22A_R1_RECOVERY.zip into ~/Downloads:
  bash "$HOME/Downloads/STEP22A_R1_RECOVERY/run.sh"
Default source is the exact interrupted evidence directory:
  ~/EdgeAI-Optimization/13_Audit_Evidence/STEP22A_FULL_PRODUCER_CAPACITY_yukf5otH
An explicit source path can be passed as the first argument, but its manifest digest must match the reviewed original. Do not pass 'full' or 'pilot' to this wrapper.
Estimated new acquisition time: about 25–35 minutes, plus source verification, copying and ZIP compression. Do not launch another GPU/CPU benchmark. tmux is recommended for SSH disconnection protection.

EVIDENCE PROTECTION
A fresh STEP22A_R1_RECOVERY_* directory is created. Original source is READ ONLY to this workflow. The original 2,501 manifest hashes and exact source run inventory are checked before acquisition. Accepted raw run folders are copied into the new tree, not linked; old metadata is retained separately. Clock apply/restore uses the prior verified logic and shared lock. Raw quality rejection remains in force; a repeated failure is preserved and reported rather than hidden.
The merged run file and final summaries contain 512 arms only on successful completion: 492 copied + 20 acquired. acquisition_sessions.csv identifies every arm's original/recovery session. protocol.json records the recovery provenance and superseded arm; original_protocol.json and STEP22A_SOURCE_SHA256SUMS.txt preserve the source metadata. summarize.py checks all 512 raw run folders and independently reintegrates their energy before completion. This merged evidence is multi-session, which must be disclosed in the manuscript; pairing is preserved but session/environmental equivalence is not established merely by hashes.
The new STEP22A_R1_REVIEW.zip contains all accepted raw folders and new logs. Keep the original STEP22A_REVIEW.zip and its evidence directory: the superseded/failed original arms are preserved there.

VALIDATION
Local checks: all source hashes, 493 completed raw traces, replay/order/count checks, deterministic 20-arm recovery plan, paired-retention tests, invalid-source rejection, unchanged rejection of >150 ms power gaps, Python and shell syntax. This recovery package has NOT run on Jetson yet. New outputs are pending; no recovered scientific results are claimed.
