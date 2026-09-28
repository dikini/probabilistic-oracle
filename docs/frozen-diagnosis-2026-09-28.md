# Frozen diagnosis: direct answers versus external Bayes

With frozen Laya, supplied probability tables did not make direct answers or
model-estimated probability components reliable. Both performed worse than a
prior-only baseline. Extracting symptoms and using the provided exact tables
externally gave better log loss, but extraction errors limited that hybrid too.

This experiment separates three sources of error: incorrect parameter estimates,
incorrect symptom extraction, and an incorrect independence assumption. Exact
controls demonstrate the structural effect even though model errors dominate
several measured pipelines. No neural training or probability fitting occurred.

## Design and information boundaries

The [protocol](plans/2026-09-28-frozen-diagnosis.md) was frozen in `19a9d1d`;
inference used clean source `d7ec510`. The checkpoint/runtime are the same pinned
English Laya setup as the [previous evaluation](laya-oracle-2026-09-28.md): Laya
0.3.21, model revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, CUDA with
BF16 autocast, no routing, compilation or model training. Raw decision logits
and complete SDK responses were saved. Shipped temperature scaling is primary;
raw softmax is reported separately. No new calibration was fitted.

Every machine has one of three fictional faults: coolant, bearing or sensor,
with probabilities 0.5, 0.3 and 0.2. X is high temperature; Y is vibration.
Conditional X probabilities are 0.8, 0.4, 0.1; conditional Y probabilities are
0.7, 0.3, 0.2. Two worlds have these same priors and marginals:

- Independent: P(X,Y|fault) is the product of the conditional marginals.
- Correlated: P(X=1,Y=1|fault) is 0.68, 0.28, 0.09; other cells preserve marginals.

Full information supplies priors, marginal probabilities and joint symptom tables.
Reduced information omits only the joint cells. Inputs never reveal which world
they belong to. All 66 pairs of reduced-information requests across worlds had
identical text, questions and observed logits. Different scores then reflect the
world's target distribution, not hidden world information reaching the model.

Four symptom patterns are fully enumerated, with three phrasings and two option
orders: 96 diagnosis presentations across worlds and information conditions.
These represent eight underlying world/pattern cases, not 96 independent samples.
Scoring integrates over the true fault and symptom distributions, then averages
presentations equally. Accuracy is expected correctness of the predicted argmax;
log loss is expected negative log probability, in nats. It does not force an
ambiguous symptom pattern to have one deterministic true fault. Multiclass Brier
is the sum over the three classes. Log loss uses a 1e-12 numerical floor; no
reported predictions contained zero probabilities.

## What each pipeline receives and computes

| Pipeline | Laya supplies | External computation |
| --- | --- | --- |
| Direct | Three-way final diagnosis distribution, from table and report | Scoring only |
| Factorized joint | Prior and one joint symptom distribution per fault, plus extracted symptom state | Exact Bayes with model-supplied parameters |
| Factorized naive | Prior and two marginal symptom likelihoods per fault, plus extracted state | Product of likelihoods, then Bayes |
| Hybrid joint | Extracted symptom state | Bayes with the supplied exact prior and joint table |
| Hybrid naive | Extracted symptom state | Bayes with supplied prior and marginal products |

Direct and parameter-estimation questions have the same available probability
table. Parameter questions exclude the case report: the prior is pre-evidence,
and likelihoods condition on a specified fault. One extraction response is shared
between factorized and hybrid paths. Parameters are estimated once per world,
coverage, phrasing and order, then reused for all four reports. Joint methods are
unavailable when the joint table is removed; no hidden joint parameters are fed
to a reduced-information competitor.

Hybrid methods receive exact numeric parameters externally. They therefore test
the value of splitting extraction from calculation, not whether a learned Bayesian
model would recover those parameters. Known versus CPU-estimated parameters was
not compared; the known-parameter option was used in this first experiment.

## Full-information results

Higher expected accuracy is better; lower log loss and Brier are better.

| Method | Independent accuracy | Independent log loss | Correlated accuracy | Correlated log loss |
| --- | ---: | ---: | ---: | ---: |
| Direct | 34.43% | 1.2658 | 35.00% | 1.2563 |
| Factorized joint | 35.33% | 1.3089 | 35.17% | 1.2688 |
| Factorized naive | 25.80% | 1.3554 | 29.47% | 1.3233 |
| Hybrid joint | 51.60% | 1.0038 | 46.33% | 1.0075 |
| Hybrid naive | 51.60% | 1.0038 | 52.37% | 1.0184 |
| Prior only | 50.00% | 1.0297 | 50.00% | 1.0297 |
| Exact joint, true symptoms | 61.40% | 0.7834 | 59.60% | 0.8445 |
| Exact marginals, true symptoms | 61.40% | 0.7834 | 56.80% | 0.9058 |

The decomposition into learned probability components did not improve log loss
in this run. Even with the *true* symptom state substituted, factorized joint
log loss was 1.3226 (independent) and 1.2648 (correlated), still worse than the
1.0297 prior-only baseline. This isolates substantial parameter/readout error
beyond symptom extraction.

The hybrid improved log loss over both direct and factorized pipelines. However,
correlated hybrid-joint accuracy was below prior-only accuracy. Hybrid naive had
higher accuracy and lower Brier than hybrid joint, while joint had slightly better
log loss. These different rankings, with faulty extracted evidence, must not be
interpreted as proof that assuming independence is preferable.

Shipped scaling helped direct and factorized distributions relative to raw logits,
but did not reverse the main failure. Raw direct log loss was 1.5656/1.5593 and
raw factorized-joint log loss was 1.6534/1.5717 in independent/correlated worlds.

## Removing the joint table

| Method | Independent accuracy | Independent log loss | Correlated accuracy | Correlated log loss |
| --- | ---: | ---: | ---: | ---: |
| Direct | 24.60% | 1.4661 | 25.37% | 1.4899 |
| Factorized naive | 20.53% | 1.3158 | 21.10% | 1.3082 |
| Hybrid naive | 51.60% | 1.0038 | 52.37% | 1.0184 |
| Prior only | 50.00% | 1.0297 | 50.00% | 1.0297 |
| Exact marginals, true symptoms | 61.40% | 0.7834 | 56.80% | 0.9058 |

Direct prediction worsened after table removal. Factorized-naive log loss improved
slightly instead, despite still being worse than prior-only. This suggests that
supplying more correct numbers is not monotonically helpful for this frozen
interface. Hybrid naive is unchanged: it uses the same extracted symptoms and
marginal parameters in both conditions.

In the correlated world, reduced information cannot identify the correct joint
likelihoods. The exact-joint score remains a hidden-truth reference, not an
available method in that condition. A fair conclusion is information loss plus
model limitations, not that a model should reconstruct an undisclosed dependency.

## Isolating dependence and double-counting

With exact parameters and true symptom labels, joint and naive computations are
equal in the independent world. In the correlated world, naive log loss is
0.9058 versus 0.8445 for the joint model: **0.0613 nats worse**. Expected accuracy
falls from 59.6% to 56.8%. This difference is entirely structural; no model output
or extraction error is involved.

For both symptoms present, the exact correlated posterior over
(coolant, bearing, sensor) is (0.7692, 0.1900, 0.0407). Treating correlated symptoms
as independent gives (0.8750, 0.1125, 0.0125), overconfidently favoring coolant.
For normal temperature with vibration, exact joint Bayes favors sensor at 0.5789;
naive Bayes assigns sensor only 0.2250 and instead favors coolant. Dependence can
change both confidence and the selected fault.

## Extraction and parameter failures

Laya extracted the correct pair in **14/24** unique report presentations. All six
normal/no-vibration reports were correct; normal/vibration was correct 4/6,
high/no-vibration 1/6, and high/vibration 3/6. For example, reversing options changed
'Temperature is high and vibration is absent.' from the correct high/no-vibration
state to normal/no-vibration. Every input was verified untruncated, and captured
logits reproduce SDK probabilities and choices, so this is an observed model/
interface error rather than a missing-probability or parser fallback.

Population-weighted extraction accuracy was 60.33% in the independent world and
68.47% in the correlated world; the same extraction responses were reused, but
symptom-pattern frequencies differ. The earlier 119/120 support-record result does
not generalize to this joint-symptom extraction task.

The true prior is always (0.5, 0.3, 0.2). Mean shipped estimates across presentations
were approximately (0.366, 0.116, 0.517) with the independent full table,
(0.445, 0.099, 0.457) with the correlated full table, and (0.239, 0.246, 0.515)
with either reduced table. The table's explicit prior was not reliably recovered.
Mean absolute errors of full-table marginal likelihood estimates were 0.2634 and
0.2641; joint-cell errors were 0.1919 and 0.2057. These are descriptive parameter
errors; they are not a new calibration fit.

## Sensitivity and inference cost

Total variation here is half the sum of absolute differences in the final fault
distribution. Each maximum spans three phrasings and two option orders for the
same underlying symptom pattern. It includes both kinds of presentation change.

| Full-information method | Maximum TV, independent | Maximum TV, correlated | Requests per 24 diagnosis presentations | Parameter requests included |
| --- | ---: | ---: | ---: | ---: |
| Direct | 0.3995 | 0.4081 | 24 | 0 |
| Factorized joint | 0.3022 | 0.2407 | 48 | 24 |
| Factorized naive | 0.3712 | 0.2174 | 66 | 42 |
| Hybrid joint | 0.4455 | 0.5585 | 24 | 0 |
| Hybrid naive | 0.4455 | 0.4455 | 24 | 0 |

For the independent/full condition, measured SDK-call sums were 0.878 seconds for
direct, 1.617 for factorized joint, 2.193 for factorized naive, and 0.703 for hybrid.
The factorized sums include parameter setup (0.914 and 1.490 seconds); the rest is
shared extraction. Across many future reports from the same population, those
parameter calls could be amortized further. Shared calls are counted within each
method's cost, not added repeatedly for each report. Do not add these totals across
methods as if every shared observation were a separate physical call.

These are one-process, sequential GPU timings, not a throughput benchmark. First-
call warmup and order effects can affect individual times. Both hybrid variants
share the same model call. Their external arithmetic is not included in the SDK
sum; recomputing the entire two-readout report took about 0.07 seconds on CPU.

The actual combined run made 324 calls in 21.7 seconds including model startup;
SDK calls summed to 11.36 seconds. Peak sampled total GPU memory was 3,672 MiB
on the RTX 4050 laptop shared with the GUI. The process exited normally, and no
compute worker remained. No neural training, parameter fitting or GPU expansion
was needed.

## Conclusion

This run does not support using frozen Laya's direct distribution or its estimated
probability components as a reliable diagnosis oracle, even with the needed
numbers supplied. Explicit Bayes works correctly with known parameters and
correct evidence; its structure matters when symptoms are dependent. In the actual
hybrid pipeline, evidence extraction is an additional bottleneck that can reverse
accuracy rankings and must be evaluated separately.

The most useful next frozen-model comparison would test simpler single-symptom
extraction and repeat the same fixed diagnostic suite with Qwen3-4B-AWQ. That would
separate this Laya interface's limitations from a general claim about factorization.
Neither follow-up was performed here. Marginal likelihood and Occam-style model
selection remain deferred until these predictive and extraction issues are better
understood.

## Verification and artifacts

CPU verification: **146 passed, one optional vLLM GPU integration test skipped**.
Tests include world normalization, equal marginals, independent equivalence,
correlated divergence, reduced prompt equality, no case-report leakage into
parameters, expected scoring, incomplete records, saved-world drift rejection,
and a synthetic perfect-parameter fixture that makes all exact joint pipelines
agree. That fixture is explicitly a CPU test, not reported model evidence.

The report reproduced exactly from all 324 real raw records; all remained on CUDA.
Longest model input was 328 of 512 tokens. SDK token usage matched saved sequences,
all options remained distinct, and every distribution passed the logit/SDK
consistency checks. Review identified a reporting reproducibility gap: saved worlds
were recorded but not validated against the current generator. A post-run guard
now rejects any mismatch rather than silently rescoring a run against changed
truth. It leaves the observed results unchanged.

Ignored artifacts: `runs/frozen-diagnosis-20260928/{manifest.json,records.jsonl,
report.json}`, adjacent `.log` and `-gpu.json`. The report contains every predicted
posterior, target, expected score and request provenance for both raw and shipped
readouts, including true-evidence factorized ablations. The complete generator and
runner are in [scripts/frozen_diagnosis.py](../scripts/frozen_diagnosis.py).

Request SHA-256:
`cc34f51343390566041376591c5b27778eb1ac52a1e30c3a2c13741b25f5be91`.

```bash
# Use the pinned environment from requirements-laya.lock and a future ISO deadline.
USE_TF=0 HF_HUB_OFFLINE=1 .venv-laya/bin/python scripts/frozen_diagnosis.py \
  --output runs/new-frozen-diagnosis --deadline "$RUN_DEADLINE"

# CPU-only report recomputation:
.venv/bin/python scripts/frozen_diagnosis.py \
  --output runs/frozen-diagnosis-20260928 --report-only
```

Use a new output directory for inference. Offline mode requires the previously
downloaded pinned snapshot. The original run also had a ten-minute process-group
watchdog, never reached. Code and report are committed; raw runs and weights remain
outside Git.

## Complete primary metric table

The exact-joint row in reduced conditions is a hidden-truth reference only.

| World | Information | Method | Accuracy | Log loss | Brier |
| --- | --- | --- | ---: | ---: | ---: |
| independent | full | direct | 0.3443 | 1.2658 | 0.7659 |
| independent | full | factorized_joint | 0.3533 | 1.3089 | 0.7893 |
| independent | full | factorized_joint_oracle_evidence | 0.3533 | 1.3226 | 0.8006 |
| independent | full | factorized_naive | 0.2580 | 1.3554 | 0.8120 |
| independent | full | factorized_naive_oracle_evidence | 0.2233 | 1.3751 | 0.8309 |
| independent | full | hybrid_joint | 0.5160 | 1.0038 | 0.5877 |
| independent | full | hybrid_naive | 0.5160 | 1.0038 | 0.5877 |
| independent | full | prior_only | 0.5000 | 1.0297 | 0.6200 |
| independent | full | reference_joint | 0.6140 | 0.7834 | 0.4702 |
| independent | full | reference_naive | 0.6140 | 0.7834 | 0.4702 |
| independent | reduced | direct | 0.2460 | 1.4661 | 0.9143 |
| independent | reduced | factorized_naive | 0.2053 | 1.3158 | 0.8163 |
| independent | reduced | factorized_naive_oracle_evidence | 0.2107 | 1.3035 | 0.8056 |
| independent | reduced | hybrid_naive | 0.5160 | 1.0038 | 0.5877 |
| independent | reduced | prior_only | 0.5000 | 1.0297 | 0.6200 |
| independent | reduced | reference_joint | 0.6140 | 0.7834 | 0.4702 |
| independent | reduced | reference_naive | 0.6140 | 0.7834 | 0.4702 |
| correlated | full | direct | 0.3500 | 1.2563 | 0.7519 |
| correlated | full | factorized_joint | 0.3517 | 1.2688 | 0.7346 |
| correlated | full | factorized_joint_oracle_evidence | 0.3517 | 1.2648 | 0.7311 |
| correlated | full | factorized_naive | 0.2947 | 1.3233 | 0.7584 |
| correlated | full | factorized_naive_oracle_evidence | 0.2753 | 1.3350 | 0.7738 |
| correlated | full | hybrid_joint | 0.4633 | 1.0075 | 0.6149 |
| correlated | full | hybrid_naive | 0.5237 | 1.0184 | 0.5955 |
| correlated | full | prior_only | 0.5000 | 1.0297 | 0.6200 |
| correlated | full | reference_joint | 0.5960 | 0.8445 | 0.5054 |
| correlated | full | reference_naive | 0.5680 | 0.9058 | 0.5313 |
| correlated | reduced | direct | 0.2537 | 1.4899 | 0.9296 |
| correlated | reduced | factorized_naive | 0.2110 | 1.3082 | 0.8111 |
| correlated | reduced | factorized_naive_oracle_evidence | 0.2220 | 1.3114 | 0.8109 |
| correlated | reduced | hybrid_naive | 0.5237 | 1.0184 | 0.5955 |
| correlated | reduced | prior_only | 0.5000 | 1.0297 | 0.6200 |
| correlated | reduced | reference_joint | 0.5960 | 0.8445 | 0.5054 |
| correlated | reduced | reference_naive | 0.5680 | 0.9058 | 0.5313 |
