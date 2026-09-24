# Counterbalanced category letters and display order

Counterbalancing reveals large formatting effects, but does not rescue the
probability readout. The earlier TRUE/A/first collapse was not a simple always-A
rule: after independently rotating the category-to-letter mapping and display
order, C and the midpoint category were favored overall, with substantial
position effects. Averaging layouts improved raw accuracy but the frozen
calibrated averages still failed to beat a constant predictor.

The two deterministic controls were barely distinguished despite different truth
values. Their raw balanced scores were both near 0.60. This strengthens concern
about categorical instruction following or task interpretation in this model/setup,
without identifying one sole cause or establishing general model incapability.

## Design and execution

The [prospective protocol](plans/2026-09-24-counterbalanced-readout.md), runner and
tests were committed as `3fcf175` before inference, with a clean working tree.
Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca` ran in BF16
through the existing vLLM 0.11.2 backend, thinking disabled, greedy one-token
output, full-vocabulary raw log-probabilities. Model weights were unchanged.

Each proposition used every crossed cyclic mapping and order rotation: nine
layouts for three categories, 25 for five. Each category occupied every
letter/position pair exactly once. The zero-shift layout matched the preceding
categorical prompt exactly. The model saw categorical meanings and letter answers,
without numerical anchors or a request for probabilities. Letter logits were
mapped into semantic category order before scoring.

Four fresh finite-card worlds (seed 2026092404), excluding all earlier count
tables, supplied H, not-H and H-and-E base propositions: 12 synthetic problems.
Two additional controls stated that all 100 cards are red, and asked whether the
selected card is red (true) or blue (false). The 14 problems yielded **476
requests**, shuffled deterministically across problems and layouts. Repeated
layouts are not independent problems.

All requests succeeded. The complete report was independently recomputed from raw
records and matched exactly. The process exited normally after **586.1 seconds**,
within the eleven-minute limit. Total sampled GPU memory peaked at **4,300 MiB**
and returned to **1,144 MiB** after exit. All 476 tokenizer-boundary checks passed;
the longest checked prompt including generation used 136 of 512 tokens.

No calibration was fitted on these cases. Secondary calibrated scores used the
previous experiment's frozen parameters: T=8, bias=-0.5 for three categories;
T=2, bias=-1 for five. Each layout was transformed separately before averaging.

## What was selected

The following counts cover the 12 synthetic problems only: 108 three-category
requests and 300 five-category requests. They are raw-logit argmax choices.

| Coordinate | Three categories | Five categories |
| --- | --- | --- |
| Category | TRUE 43; midpoint 50; FALSE 15 | TRUE 81; MOSTLY TRUE 84; midpoint 121; MOSTLY FALSE 3; FALSE 11 |
| Letter | A 43; B 14; C 51 | A 52; B 49; C 126; D 29; E 44 |
| Display position | 1st 50; 2nd 15; 3rd 43 | 1st 92; 2nd 0; 3rd 112; 4th 64; 5th 32 |

There were six exact top-logit ties for three categories and 24 for five. As
prespecified, ties favor the first category in semantic order, so argmax counts
need that qualification. Candidate-normalized mean weights avoid the tie-breaking
artifact: C received **0.4371** and **0.3310** on the three- and five-category
scales; the midpoint received **0.5275** and **0.3059**, respectively. Uniform
weights would be 1/3 and 1/5. These weights are token-distribution diagnostics,
not calibrated probabilities that those categories are true.

| Conditional choice | Three | Five |
| --- | ---: | ---: |
| A chosen when A is not first | 19/72 (26.4%) | 16/240 (6.7%) |
| First option chosen when it is not A | 26/72 (36.1%) | 56/240 (23.3%) |
| TRUE chosen when neither A nor first | 12/48 (25.0%) | 61/192 (31.8%) |

These observations reject a literal always-A, always-first, or always-TRUE account
of the previous collapse. They show mixed dependencies on category wording,
letters, and placement. Cyclic layouts do not test every permutation or fully
separate higher-order interactions; the results should not be reduced to one
universal bias coefficient.

Allowed candidate letters retained mean full-vocabulary mass **0.99916** (three)
and **0.99854** (five), and every synthetic generated token was allowed. Missing
candidate probability or gross output-format failure does not explain the result.

## Sensitivity and accuracy

Changing only the answer mapping and order produced an average per-problem raw
score standard deviation of **0.1500** (three) and **0.1383** (five). The average
within-problem max-minus-min ranges were **0.5963** and **0.5724**. These are changes
to the same proposition, not differences between worlds.

RMSE below is the mean of four world-level RMSEs, each using three synthetic
propositions. Controls are excluded. This smaller diagnostic differs from the
previous benchmark and must not be compared numerically across datasets.

| Score | Three-category RMSE | Five-category RMSE |
| --- | ---: | ---: |
| Raw, original layout | 0.306646 | 0.360215 |
| Raw, averaged layouts | 0.251488 | 0.266170 |
| Frozen calibration, original layout | 0.202342 | 0.206048 |
| Frozen calibration, averaged layouts | 0.201619 | 0.203520 |
| Constant 0.5 | 0.214202 | 0.214202 |
| Previous calibration-mean constant (0.431496) | 0.201078 | 0.201078 |

Averaging reduced raw error, but neither frozen calibrated average beat the
previous calibration-mean constant. Some improvement is compatible with cancelling
format biases and shrinking scores toward the middle; it is not evidence of
reliable probability reasoning. No full benchmark gate verdict is claimed:
conditionals, context variants and fresh calibration were intentionally omitted.

## Deterministic controls

The context was: all 100 cards in the deck are red; one is selected uniformly.
The true proposition says the selected card is red; the false one says blue.

| Readout | True control raw average | False control raw average | Difference |
| --- | ---: | ---: | ---: |
| 3 categories | 0.612873 | 0.597264 | 0.015609 |
| 5 categories | 0.624505 | 0.596216 | 0.028290 |

The correct endpoint category won only 4/9 layouts for the three-category true
control and 1/9 for the false control. With five categories, the counts were
8/25 and 1/25. Layout changes moved these control scores across ranges of
0.4567–0.4820, much larger than their averaged true-versus-false separation.
These are two simple controls, not a general capability evaluation, but they
show that formatting alone is not the only unresolved problem.

## Evidence and limits

The next investigation should test whether the model reliably follows semantic
category definitions on elementary determinate propositions under a different
instruction formulation or a stronger model. More calibration on these cases
would not establish that capability. Any confirmatory test needs fresh examples.

The runner keeps raw observations, mappings, prompts/token IDs, revisions,
settings, hashes and prior calibration provenance in ignored local artifacts:
`runs/counterbalanced-readout-20260924/{manifest.json,records.jsonl,report.json}`,
with adjacent `.log` and `-gpu.json` files. Incomplete, duplicate, failed or
inconsistent observations are rejected. The report checks the full crossed grid,
semantic mappings, prompt text, requests, provenance, raw scores/mass and hashes.

Recompute without GPU inference from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/counterbalanced_readout.py \
  --output runs/counterbalanced-readout-20260924 --report-only
```

CPU verification: **112 tests passed, one optional GPU integration test skipped**.
The GPU experiment independently exercised all 476 actual requests. Code review
also checked synthetic always-A, always-first and always-TRUE patterns.

- Query SHA-256: `83f6bfc2b22e4c72a3c5e7eb756d320aa4b76967d4f86a7b506dd30b3e605150`
- Protocol SHA-256: `245a25bd8ea7799ccff6a79ced31e48ae19d0b036fb5ac272f7b0733b26bbe1b`
- Frozen calibration SHA-256: `401b714024be5582f39fe715e810287f79aa737135d9d4a26a7b82d866d5f3ad`
