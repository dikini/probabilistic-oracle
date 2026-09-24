# Three- and five-category truth-scale results

Adding categorical answers did not produce a useful probability signal in this
setup. Both categorical scales failed the accuracy and conjunction gates, and
neither beat the calibration-mean constant baseline. The five-category scale was
slightly worse than binary on these fresh worlds. Calibration mostly compressed
scores near the mean calibration target.

## Protocol and execution

The [prospective protocol](plans/2026-09-24-categorical-readout.md) was committed
before inference. Code and tests were committed as `f32d54b`; the worktree was
clean when the manifest was written. The experiment reused Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, BF16 through vLLM 0.11.2, thinking
disabled, one-token greedy generation, raw full-vocabulary log-probabilities.
The RTX 4050 laptop remained shared with the GUI.

The model received categorical instructions with letter answers:

| Scale | Categories, from A onward | External anchors |
| --- | --- | --- |
| Binary control | TRUE; FALSE | 1; 0 |
| Three | TRUE; NEITHER TRUE NOR FALSE; FALSE | 1; 0.5; 0 |
| Five | TRUE; MOSTLY TRUE; NEITHER TRUE NOR FALSE; MOSTLY FALSE; FALSE | 1; 0.75; 0.5; 0.25; 0 |

The prompts did not ask for probabilities or expose the anchors. Each letter was
verified as one token at the actual prompt boundary. For category log-probability
l_i and external anchor v_i, the derived score was the weighted mean of anchors
under softmax(l_i/T + b*v_i). Raw scores used T=1, b=0. Temperature and ordinal
bias were fitted separately for each scale using the same fixed grid and
soft-target log-loss objective.

Four fresh calibration worlds supplied 24 base queries per scale. Eight fresh
evaluation worlds supplied 144 queries per scale, including base, paraphrased,
and irrelevant-context variants. Seed 2026092403 and explicit exclusion of every
previously used count table separated the new cases from earlier experiments.
Calibration parameters were saved before evaluation began. No parameters, prompts,
parsers, or category choices were changed after seeing the holdout.

All **504 requests** produced valid observations. The report was saved and then
independently recomputed from raw observations with an exact match. At 600.2
seconds, the external watchdog terminated the process during shutdown after all
requests and reporting had finished (supervisor exit code -15). This was not an
interrupted evaluation. Total sampled GPU usage peaked at **4,303 MiB** and
returned to **1,099 MiB**; no GPU compute process remained.

## Accuracy

RMSE is the mean of per-world RMSE across six base-query targets. Lower is better.

| Readout | Raw RMSE | Calibrated RMSE | Fitted T | Fitted bias |
| --- | ---: | ---: | ---: | ---: |
| 2 categories | 0.371665 | 0.222240 | 4 | -0.5 |
| 3 categories | 0.355982 | 0.223050 | 8 | -0.5 |
| 5 categories | 0.365905 | 0.225689 | 2 | -1 |

The constant-0.5 baseline achieved RMSE **0.246100**. The calibration-mean
constant (0.4314955646) achieved **0.222998**. Binary scores marginally beat that
constant; neither categorical scale did. All three missed the **0.10** accuracy
requirement by a wide margin.

Paired calibrated RMSE differences against binary, with descriptive 95% whole-world
bootstrap intervals (500 resamples, seed 0):

- Three categories: **+0.000810**, interval **[-0.001508, +0.003342]**.
- Five categories: **+0.003450**, interval **[+0.001798, +0.005216]**.

Positive means worse than binary. The three-category difference is unresolved;
the five-category interval is above zero in this small exploratory comparison.
There are eight independent evaluation worlds and two comparisons, with no
multiplicity adjustment, so this is not a broad ranking of elicitation methods.

## Consistency and stability after calibration

| Metric | Binary | Three | Five | Gate |
| --- | ---: | ---: | ---: | ---: |
| negation error | 0.163517 | 0.129660 | 0.083450 | 0.10 |
| conjunction error | 0.242662 | 0.243443 | 0.264562 | 0.10 |
| bayes error | 0.013520 | 0.007625 | 0.044966 | 0.10 |
| paraphrase mae | 0.008762 | 0.008096 | 0.021399 | 0.05 |
| irrelevant context mae | 0.015168 | 0.010103 | 0.018751 | 0.05 |

All Bayes paths were defined. Three categories additionally failed negation
consistency; five passed it. Good stability and Bayes-path agreement did not
compensate for poor probability accuracy and conjunction consistency.

## What the categories revealed

The raw highest-scoring category was TRUE on **144/144** evaluation prompts for
both binary and three-category readouts. With five categories it was TRUE on
**138/144**, and NEITHER TRUE NOR FALSE on **6/144**. Neither MOSTLY category nor
FALSE was the highest-scoring category. These counts include context variants,
not 144 independent worlds.

This is not missing candidate mass: mean full-vocabulary probability assigned to
the allowed letters was 99.81% (binary), 99.95% (three), and 99.87% (five). Every
generated evaluation token belonged to its allowed set. The model followed the
letter-output format but did not use the requested spectrum effectively.

Calibrated evaluation scores were narrowly distributed:

| Scale | Mean score | Standard deviation | Range |
| --- | ---: | ---: | --- |
| Binary | 0.431295 | 0.010378 | 0.407333–0.453262 |
| Three | 0.440837 | 0.007777 | 0.426192–0.458013 |
| Five | 0.436010 | 0.020273 | 0.393662–0.486597 |

These descriptive statistics cover all 144 evaluation variants per scale. The
three-category fit selected the largest temperature in the prespecified grid,
which flattens category weights. The observed accuracy is close to a constant
prediction rather than evidence of a well-calibrated semantic oracle.

## Interpretation and reproducibility

This experiment does not support adding more answer categories as a remedy for
the tested model and prompts. It also exposes a useful unresolved diagnostic:
TRUE was always listed first and mapped to A. The experiment cannot distinguish
truth bias from letter/position bias. A counterbalanced category-to-letter mapping
on fresh cases would test that explanation. The midpoint wording also mixes
intermediate truth and uncertainty. Binary uses its earlier prompt wording, so
the comparison measures prompt plus scale rather than category count alone.

The runner remains a standalone experimental script; the core binary API is
unchanged. Raw local artifacts are ignored by Git under the existing policy:
`runs/categorical-readout-20260924/{manifest.json,records.jsonl,calibration.json,report.json}`,
plus adjacent `.log` and `-gpu.json` files. Reporting validates query/protocol
hashes, raw scores/masses, candidate presence, requests, provenance, scale metadata,
and the frozen calibration. Missing or failed observations cause a fail-closed
exception rather than a success report.

Recompute from the repository root without loading a model:

```bash
PYTHONPATH=src .venv/bin/python scripts/categorical_readout.py \
  --output runs/categorical-readout-20260924 --report-only
```

CPU verification: **95 tests passed; one optional GPU integration test skipped**.
The actual experiment separately exercised the GPU backend for all 504 requests.

- Query SHA-256: `19818df7fdd6f336fd6866c87053e1c4147b41637851ace9a6837381b37f57aa`
- Protocol SHA-256: `33bcfe6ae10119cb2452c59eb884c2fdc9c853e08d5063996e13cfcb069fd649`
