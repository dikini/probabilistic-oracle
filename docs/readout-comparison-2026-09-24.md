# Numerical answers versus binary-token scores

The same Qwen3-0.6B model failed the accuracy target with both readouts on fresh
problems. Numerical answers improved the point estimate slightly, but the paired
interval includes zero. This does not support a diagnosis that binary-token
readout alone caused the earlier failures. Model capability and task formulation
remain stronger candidates for investigation, without establishing general model
incapability.

## Frozen protocol and execution

The [prospective protocol](plans/2026-09-24-readout-comparison.md), runner, and tests
were committed as `49e1667` before inference. Eight fresh card-count worlds
(seed 2026092402) excluded every table used in the initial benchmark and sweep.
Six probability queries and three context variants per world produced 144 matched
problems and 288 requests. Both readouts received identical context, propositions,
assumptions, and observations; their answer instructions differed. Request order
alternated.

The pinned model was `Qwen/Qwen3-0.6B` at
`c1899de289a04d12100db370d81485cdf75e47ca`, BF16 through vLLM 0.11.2, thinking
disabled. Numerical generation was greedy with at most 16 output tokens. The
strict parser accepted only a complete number in [0, 1] and a normal stop.
No repairs, prompt changes, or calibration fits followed inspection of results.
Binary scoring reused A/B and the previous calibration (temperature 2, bias -0.5).

All 144 numerical answers and all 144 binary observations were valid. All eight
worlds entered the comparison; none were excluded. The process took 212.3 seconds,
including model startup. Total sampled GPU usage peaked at 4,364 MiB and returned
to 1,147 MiB after exit. The run stayed within its time limit.

## Accuracy and consistency

RMSE below is the mean of per-world RMSE across the six base queries. Consistency
and context-sensitivity values use the existing benchmark definitions. Lower is
better. Thresholds were fixed before this run.

| Metric | Numerical | Binary calibrated | Binary raw | Threshold |
| --- | ---: | ---: | ---: | ---: |
| Probability RMSE | 0.247129 | 0.256953 | 0.359937 | 0.10 |
| Negation error | 0.146250 | 0.072026 | 0.339370 | 0.10 |
| Conjunction error | 0.087613 | 0.236106 | 0.185080 | 0.10 |
| Bayes-path error | 0.161142 | 0.019506 | 0.030861 | 0.10 |
| Paraphrase MAE | 0.092396 | 0.018832 | 0.031735 | 0.05 |
| Irrelevant-context MAE | 0.024583 | 0.027606 | 0.045275 | 0.05 |

The constant-0.5 baseline had RMSE 0.263001. The frozen calibration-target mean
(0.47336894586894585) had RMSE 0.257336. Numerical and calibrated binary scores
both beat these baselines in point estimates, but neither approached the 0.10
accuracy requirement. Neither readout passed all gates. All Bayes paths were
defined.

Numerical minus calibrated-binary RMSE was **-0.009824**, with a paired 95%
bootstrap interval **[-0.046220, 0.026211]** (500 whole-world resamples, seed 0).
Eight independent worlds provide limited precision: the observed advantage is
not a demonstrated reliable difference. The runner's descriptive decision is
`numeric_improves_but_fails`; this describes the point estimate, not statistical
certainty.

Numerical answers particularly improved joint-event accuracy, while negative
conditioning became worse:

| Query kind | Numerical RMSE | Calibrated binary RMSE |
| --- | ---: | ---: |
| H | 0.187883 | 0.179380 |
| Not H | 0.185708 | 0.185784 |
| E given H | 0.245508 | 0.263415 |
| E given not H | 0.384488 | 0.308511 |
| H after observing E | 0.296851 | 0.265547 |
| H and E | 0.142960 | 0.325900 |

These per-kind RMSEs pool eight base queries each, so they do not average directly
to the mean per-world RMSE above. The numerical conjunction-consistency improvement
comes with worse negation, Bayes, and paraphrase behavior.

For a concrete error, the first world contains 17 red circle cards, 29 red
non-circle cards, 45 blue circle cards, and 9 blue non-circle cards. Given that
the card is not red, the circle probability is 45/54 = 0.833333. The numerical
answer was `0.35`; calibrated binary scoring gave 0.468791. For the red-and-circle
joint event, the correct probability is 0.17; the answers were `0.35` and 0.468791.
This is a descriptive example, not an additional test or selected evaluation set.

## Interpretation and evidence

Explicit numerical elicitation did not rescue this model under the tested
no-thinking prompts. Its format compliance was perfect, but its probability
estimates and identities remained unreliable. The next useful discrimination is
between task interpretation and model capability, rather than another calibration
search on these same worlds. Any further confirmatory comparison needs fresh cases.

This comparison changes instruction wording and autoregressive output length as
well as answer representation. It cannot isolate readout as a sole causal factor,
nor establish open-world calibration or the capability of larger models.

Raw artifacts are retained locally in
`runs/readout-comparison-20260924/{manifest.json,pairs.jsonl,report.json}`, with
adjacent `.log` and `-gpu.json` files. They are ignored by Git under the existing
run-artifact policy. The saved report was independently recomputed from raw
observations and matched exactly, including validation of numerical text, token
log-probabilities, calibration, requests, provenance, and manifest hashes.

- Query SHA-256: `5f8f0fae61d6728297900084cf4140482b9b06f20499eaa5cfc301fb6a1a84a4`
- Protocol SHA-256: `9400b92d0bb1894482d40fd86cd1c065e160feaa64151ebd9f2e08a21a232b64`

CPU verification before inference: 73 tests passed; one optional GPU integration
test was skipped. The actual matched experiment exercised the GPU backend for
all 288 requests.
