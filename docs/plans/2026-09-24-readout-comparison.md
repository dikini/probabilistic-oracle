# Numerical versus binary readout experiment

**Goal:** Test whether the same frozen model estimates probabilities more accurately
when asked to generate a number than when its binary answer-token scores are read.

**Architecture:** An experimental script reuses one vLLM engine, alternates matched
numerical and binary requests, and saves both observations under the same query ID.
CPU-only parsing and reporting preserve failures and compare complete paired worlds.

**Tech stack:** Existing pinned Qwen3-0.6B/vLLM runtime and Python standard library.

## Prospective protocol

Use eight new finite card worlds generated with seed 2026092402, excluding every
count table from the initial benchmark and parameter sweep. Each has the existing
six query kinds and three context variants: 144 matched problems, 288 requests.
Hash the cases and protocol before inference. No tuning on these new cases.

Reuse the selected baseline A/B prompt and frozen calibration temperature 2,
intercept -0.5 from the previous sweep. Report raw and calibrated binary scores.
The numerical prompt retains the same context, proposition, assumptions, and
observations, changing the requested answer into a number between 0 and 1.
Use greedy decoding, thinking disabled, and at most 16 generated tokens. Do not
fit any calibration to numeric outputs. Alternate which readout runs first.

Accept a complete, finite numerical string in [0, 1], including decimal or
scientific notation and surrounding whitespace. Reject percentages, fractions,
prose, empty responses, out-of-range values, and truncated responses. Preserve
the raw text, token IDs, finish reason, exact prompt, settings, and errors.
No repair prompts or retrospective parser relaxation after viewing outcomes.

Report format-valid coverage, per-kind errors, and the same probability and
consistency metrics as before. Compare both readouts with constant 0.5 and the
previous calibration-target mean (0.47336894586894585). Report mean per-world
RMSE and a paired bootstrap interval for numeric minus calibrated-binary RMSE
(500 whole-world resamples, seed 0). Only complete paired worlds enter paired
metrics; list excluded worlds and never declare success with missing observations.

Numeric accuracy success requires all responses valid, mean per-world RMSE <=0.10,
and beating both baselines. Apply the existing full consistency/stability gates
separately. Evidence favoring the numeric readout requires accuracy success and
a paired difference interval below zero. If both readouts fail, neither tested
elicitation is established as a usable oracle. If numeric improves but fails,
report partial improvement, not a validated oracle.

This changes instruction wording and generated output length as well as readout,
so numeric success would implicate the elicitation/readout package; it would not
uniquely identify answer representation as the cause. A small synthetic suite
cannot establish open-world calibration. Do not infer general model incapability.

## Execution

1. Add failing parser, paired-summary, leakage/freshness, and failure-coverage tests.
2. Implement `scripts/compare_readouts.py` and pass CPU tests.
3. Review and commit this protocol and runner before live inference.
4. Run the pinned model with a bounded external timeout, retaining raw records.
5. Recompute and inspect the report, document results, and commit on main.
