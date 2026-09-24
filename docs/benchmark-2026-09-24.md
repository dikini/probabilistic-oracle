# First benchmark: 2026-09-24

## Outcome

All 240 requests completed with valid observations and no request failures. The
predeclared continuation criteria were not met. Temperature scaling improved
some consistency metrics but did not beat either constant baseline in probability
error, and conjunction inconsistency increased. This result does not justify
moving to memory or semantic extraction.

This is one small-model experiment: four calibration worlds, eight evaluation
worlds, and four ordinary-world stability cases. It does not establish what
larger models or other prompt designs can do.

## Evaluation results

Values are means over eight evaluation worlds. Brackets show descriptive 95%
bootstrap intervals over whole worlds, using the predeclared 500 resamples.

| Metric | Raw | Calibrated | Continuation limit |
| --- | ---: | ---: | ---: |
| Probability RMSE | 0.2659 [0.2274, 0.3006] | 0.2499 [0.2141, 0.2830] | 0.10 |
| Soft-target log loss | 0.7114 [0.6966, 0.7264] | 0.6938 [0.6919, 0.6956] | — |
| Complement error | 0.1329 [0.1026, 0.1594] | 0.0171 [0.0132, 0.0205] | 0.10 |
| Conjunction error | 0.1778 [0.1508, 0.1997] | 0.2410 [0.2378, 0.2437] | 0.10 |
| Bayesian-path error | 0.0651 [0.0396, 0.0924] | 0.0083 [0.0051, 0.0117] | 0.10 |
| Paraphrase absolute change | 0.1012 [0.0910, 0.1114] | 0.0132 [0.0118, 0.0146] | 0.05 |
| Irrelevant-context absolute change | 0.1351 [0.1167, 0.1494] | 0.0177 [0.0154, 0.0197] | 0.05 |

| Baseline | Probability RMSE |
| --- | ---: |
| constant_half (score 0.5000) | 0.2494 |
| calibration_mean (score 0.4734) | 0.2473 |

The calibrated point estimate is slightly worse than both baselines. These small
differences are not a claim of statistically significant inferiority; failure to
beat the baselines, RMSE above 0.10, and conjunction error above 0.10 fail the
prospective engineering gates. No Bayesian paths were undefined.

## Calibration and interpretation

The calibration split selected temperature **8**, the largest candidate in the
predeclared grid. It was fitted on 24 base queries from four calibration worlds
without evaluation labels. Large temperatures pull binary scores toward 0.5.
That can reduce complement errors and prompt sensitivity without adding useful
probability information. Near-constant 0.5 scores also produce a conjunction
discrepancy near 0.25 because 0.5 differs from 0.5 × 0.5. The measured conjunction
error is consistent with that limitation.

Ordinary-world results are stability checks only: raw mean complement error
was 0.1626, paraphrase absolute change 0.2034, and irrelevant-context absolute
change 0.0473. They were excluded from temperature fitting and target-based
quality metrics. No ordinary-world calibration claim is made.

The combined TRUE/FALSE vocabulary mass averaged 0.9986 and was at least
0.9957. In this run, low candidate mass does not explain the poor target accuracy.
This remains a plausibility-scoring experiment, not a validated probability oracle.

## Runtime and evidence

- Qwen3-0.6B, pinned revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- vLLM 0.11.2, BF16, context 512, one sequence, 50% GPU memory budget.
- Wall time including engine startup and reporting: 296.2 seconds.
- Median scoring request time: 1.12 seconds, including full-vocabulary extraction.
- Sampled peak total GPU use: 4,276 MiB; minimum reported free: 1,487 MiB.
- No compute process remained in the post-run GPU check. GUI responsiveness was not measured.
- The tests passed: 45 without GPU dependencies; one separate real-GPU integration test.
- An independent code review found a request/provenance validation gap; it was
  fixed with regression tests before this benchmark. No review findings remain.

Experiment code commit: `fe6fe61abf76aecbc6dcb8706a9e5c8e41e56597`; working tree was clean.
Protocol SHA-256: `04a910a807d98306acb65cf2438e460ec280cf6c92c5a301d0171cd7725a2661`.
Query SHA-256: `de59beb58bf8daf537db057b210a930933b1ac0e82f8fdbd53081940c5f24292`.

The [protocol](experiment-protocol.md) was committed before the run. Neither
thresholds nor cases were adjusted after seeing the evaluation results.

Local raw artifacts (ignored by Git):

- `runs/milestone-1-20260924/manifest.json`
- `runs/milestone-1-20260924/records.jsonl`
- `runs/milestone-1-20260924/report.json`
- `runs/milestone-1-20260924.log`
- `runs/milestone-1-20260924-gpu.json`

Recompute the report from the repository root:

```bash
.venv/bin/probabilistic-oracle report runs/milestone-1-20260924
```

Any subsequent prompt or calibration development should use development cases
and a fresh held-out suite for its final comparison. Replication with a different
model or verbalizer remains future work.
