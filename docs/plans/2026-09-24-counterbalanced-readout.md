# Counterbalanced category-letter and display-order diagnostic

**Goal:** Separate semantic category preference from letter and list-position
preference in the earlier three- and five-category readouts.

**Architecture:** Reuse categorical scoring and the verified vLLM engine. A new
standalone runner stores all crossed rotations and decodes logits back into
semantic category order before computing scores and formatting diagnostics.

**Tech stack:** Python standard library, existing Qwen3-0.6B/vLLM stack, pytest.

## Prospective design

For n=3 and n=5, keep the previous category meanings and prompt wording. Assign
semantic category i to letter (i+m) modulo n for each mapping shift m. Display
categories in cyclic order (j+o) modulo n for each order shift o. Cross all n*n
mapping/order shifts independently for every query: each semantic category occurs
at every letter/position pair once. This is a balanced cyclic design, not all
possible permutations. Record mappings and display order explicitly. The m=o=0
cell exactly matches the preceding categorical prompt and serves as paired control.
Shuffle cell execution order deterministically (seed 2026092404).

Use four fresh finite-card worlds (seed 2026092404), excluding all count tables
from every earlier run. Query H, not H, and H-and-E in base wording only: 12
problems. Add two deterministic controls: context says all 100 cards are red;
propositions selected card is red (true) and blue (false). Total: 14*(9+25)=476
requests. These are repeated formats of 14 problems, not 476 independent cases.
Use the same pinned frozen model, conservative GPU settings, no thinking, greedy
one-token output, raw logprobs and categorical letter answers. No numerical
probability requests. Eleven-minute external inference limit; preserve incomplete
records and do not claim complete findings after a missing/failed observation.

No fitting on new cases. Load the previous categorical calibration temperatures
and ordinal biases from its saved calibration.json, and retain them as secondary
frozen transforms only. Raw diagnostics are primary. Map letter logs into semantic
order, normalize, then compute expected external anchor. For each query average
scores over all rotations; calibrated averages transform each rotation first.
Never average letter logits before mapping them into semantic order.

Report raw argmax distributions by category, letter and display position; mean
candidate-normalized mass by these three coordinates; letter-A choice when A is
not first; first-position choice when that letter is not A; TRUE choice when TRUE
is neither A nor first. Include denominators and ties (first semantic category
breaks exact ties). Report per-query score range/SD across formats and paired
canonical-versus-balanced accuracy on 12 synthetic queries, with mean per-world
RMSE and constant .5/previous calibration-mean baselines. Keep deterministic
controls separate. No full benchmark gate verdict: this omits conditionals,
context variants, and new calibration. Candidate vocabulary mass and format
compliance remain explicit.

Large preference for A despite position/meaning changes supports a letter bias;
first-position preference across letters supports an order effect; stable TRUE
preference across letters/positions supports semantic truth bias or task
interpretation. Effects can coexist. Counterbalancing yielding a flat .5 score
is cancellation of nuisance effects, not evidence of useful probability reasoning.
A follow-up should depend on these findings, not retune this dataset.

## Implementation plan

1. Add failing tests in tests/test_counterbalanced_readout.py for joint balance,
   semantic remapping, canonical-prompt identity, always-A diagnostic behavior,
   missing/duplicate observations, and metadata/logit tampering.
2. Implement scripts/counterbalanced_readout.py using categorical_readout helpers;
   validate expected request grid, mappings, requests, provenance and raw scores.
3. Run full CPU suite and real-tokenizer boundary/context preflight. Review, then
   commit code/protocol before inference. Keep all raw run artifacts outside Git.
4. Execute bounded GPU run; recompute report from saved records; report runtime,
   coverage, raw distributions, formatting sensitivity and limitations.
5. Save docs/counterbalanced-readout-2026-09-24.md, link README, test, commit and
   merge locally to main.
