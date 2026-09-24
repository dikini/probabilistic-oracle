# Bounded parameter sweep: 2026-09-24

## Outcome

No tested configuration met the fixed continuation gates. The selected
configuration also failed the fresh evaluation. The experiment process took
679.6 seconds (11 minutes 20 seconds), within the requested approximately
20-minute session budget including preparation and reporting.

The model remained Qwen3-0.6B at the previously pinned revision. No weights
were changed and no larger model was downloaded.

## Search

Four new prompt/label configurations were screened, alongside a reused
TRUE/FALSE baseline. Each new configuration used 24 calibration queries and
72 development queries. These development worlds came from the already-seen
first benchmark; they are no longer described as held-out evidence.

Each candidate used 108 temperature/intercept combinations for calibration
fitting, with an additional temperature-only comparison. Parameters were fitted
on calibration targets. Candidate selection used the largest normalized gate
violation on development cases. Separate optimistic development-only diagnostics
examined all 540 combinations across the five configurations; none met all
numeric gates. Those diagnostics were not used to alter the frozen selection.

| Configuration | Raw development RMSE | Calibration-fitted RMSE | Temperature | Bias | Lowest development-only grid RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline-true-false-reused | 0.2776 | 0.2474 | 32 | 0 | 0.2320 |
| baseline-yes-no | 0.2602 | 0.2492 | 3 | 0 | 0.2320 |
| baseline-a-b | 0.3599 | 0.2387 | 2 | -0.5 | 0.2297 |
| sample-true-false | 0.6230 | 0.2399 | 32 | -0.25 | 0.2308 |
| frequency-true-false | 0.4590 | 0.2453 | 8 | -0.25 | 0.2316 |

The A/B baseline was selected with calibration temperature 2 and bias −0.5.
The explicit-frequency A/B and few-shot example prompts were skipped to reserve
time for final evaluation. Decoding temperatures 0, 0.5, and 1 produced identical
raw candidate log-probabilities on the check request. This matches the backend
contract: raw log-probabilities are measured before sampling transformations.

## Fresh evaluation

Selection was frozen at 18:30:04 UTC before scoring eight new synthetic worlds
and four ordinary-world stability cases. The fresh evaluation comprised 168
requests. Together with 384 screening requests and three temperature checks,
the session made **555 new model requests**, all successful.

| Metric | Raw | Calibrated | Gate |
| --- | ---: | ---: | ---: |
| rmse | 0.3189 | 0.2512 | <= 0.10 |
| negation_error | 0.3220 | 0.0817 | <= 0.10 |
| conjunction_error | 0.2012 | 0.2427 | <= 0.10 |
| bayes_error | 0.0374 | 0.0224 | <= 0.10 |
| paraphrase_mae | 0.0342 | 0.0201 | <= 0.05 |
| irrelevant_context_mae | 0.0479 | 0.0289 | <= 0.05 |

Fresh-case baseline RMSE: constant 0.5 **0.2482**; calibration-target mean
**0.2492**. The selected configuration reached **0.2512**, so it did not beat
either baseline. Its conjunction discrepancy remained **0.2427**, above 0.10.
The complement, Bayesian-path, and context-stability gates passed. Both raw
and calibrated Bayesian paths were defined for every evaluation case.

These are descriptive results from eight worlds, not proof that the model or
all possible prompts lack useful probability information. The practical result
is that this bounded sweep did not rescue the tested scoring method. Improving
small consistency errors through calibration did not produce accurate probability
estimates or conjunction consistency.

## Runtime and verification

- Experiment process duration: 679.6 seconds.
- Sampled peak total GPU use: 4328 MiB; minimum reported free: 1435 MiB.
- GPU compute processes were absent after exit; GUI responsiveness was not measured.
- 49 tests passed; the optional GPU integration test was skipped in that test run.
- The live experiment itself completed all 555 GPU requests without failure.
- Review found that the sweep initially checked undefined Bayesian paths only
  after calibration. The checker and ranking were corrected and regression-tested.
  Saved results were reprocessed; the selected configuration and final decision
  were unchanged. Original observations and reports are retained.

The inference process loaded commit `2f699ca`. The gate-check correction was
made while that process was running and applied to the saved observations after
completion. Use `reviewed-final-report.json` and `reviewed-leaderboard.json` as
the checked summaries.

## Artifacts and reproduction

Protocol: [parameter-sweep-protocol.md](parameter-sweep-protocol.md).

Raw observations and summaries are retained under the ignored local directory
`runs/parameter-sweep-20260924/`. It includes the calibration and development
runs, the frozen selection, fresh cases and their hash, fresh evaluation records,
both original and reviewed reports, and the complete development-grid diagnostic.
The run log and GPU monitor are adjacent in `runs/`.

Recompute the checked summaries from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_sweep.py runs/parameter-sweep-20260924
```

To reproduce the sweep in a new directory, choose a future UTC deadline and
retain an external timeout to cover a stalled startup or in-flight request:

```bash
timeout --signal=TERM --kill-after=10s 18m env PYTHONPATH=src .venv/bin/python scripts/parameter_sweep.py \
  --output runs/repeated-sweep \
  --baseline runs/milestone-1-20260924 \
  --deadline "$(date -u -d '+17 minutes' +%Y-%m-%dT%H:%M:%S%:z)"
```

A repetition reuses this now-seen holdout and is a replication, not fresh
validation. Further tuning needs another prospectively defined holdout.
