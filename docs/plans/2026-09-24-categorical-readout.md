# Categorical truth-scale experiment

**Goal:** Compare binary, three-category, and five-category token readouts from
one frozen Qwen3-0.6B model without requesting numerical probabilities.

**Architecture:** A standalone experiment reuses the verified vLLM engine and
existing finite-world metrics. It saves raw category log-probabilities, fits only
on calibration worlds, freezes parameters before evaluation, and derives reports.

**Tech stack:** Python standard library, existing Qwen/vLLM stack and pytest.

## Design and prospective protocol

Use A/B for TRUE/FALSE as the existing binary control. For three categories use
A=TRUE, B=NEITHER TRUE NOR FALSE, C=FALSE; for five use A=TRUE, B=MOSTLY TRUE,
C=NEITHER TRUE NOR FALSE, D=MOSTLY FALSE, E=FALSE. Letters avoid multi-token
verbalizers and are validated at the actual prompt boundary. The model sees only
categorical wording, not numerical anchors or a request for probabilities.

Externally assign anchors [1, .5, 0] or [1, .75, .5, .25, 0] (binary [1, 0]).
For log-probabilities l_i and anchors v_i, define
w_i = softmax(l_i/T + b*v_i), score = sum(w_i*v_i).
Raw scores use T=1, b=0. Fit T in [.25,.5,.75,1,1.5,2,3,4,8] and b in
[-2,-1,-.5,0,.5,1,2] independently per scale, minimizing soft-target log loss
on calibration base queries. Ties prefer T closest to 1, then abs(b) closest to 0.
No per-category offsets, flexible regression, or post-evaluation selection.
Temperature here transforms observed logits, not sampling temperature.

Alternative hard-category decoding discards useful logit information. Free-form
word answers introduce multi-token comparability issues. Use weighted token
scores for this experiment; retain argmax categories and candidate mass as
secondary diagnostics. Midpoint semantics mix intermediate truth and uncertainty;
this is an elicitation experiment, not an assumption of probabilistic semantics.

Generate four new calibration and eight new evaluation worlds with seed 2026092403,
excluding all count tables from the original suite, previous sweep, and numerical
comparison. Keep every variant of each world in one split. Calibration uses six
base queries/world only (24); evaluation uses six queries and three context
variants/world (144). Three scales produce 504 requests. Rotate scale order per
query. One model engine, unchanged conservative laptop settings, thinking disabled,
greedy one-token generation, full-vocabulary raw logprobs then candidate extraction.
Use a 10-minute external inference limit and cooperative deadline. Do not silently
truncate, replace missing token observations, or change prompts after inspection.

Save cases, queries, hashes, protocol, exact prompts/token IDs, ordered categories
and anchors, model/tokenizer revisions, settings, raw logs, generated token IDs,
time, candidate mass, raw scores and errors. Freeze and save fitted parameters
before evaluation requests. Missing/failed queries make the comparison incomplete.
Recompute calibration and scores from saved raw observations when reporting.

Report raw and calibrated mean per-world RMSE, soft-target log loss, consistency,
stability, candidate mass, and argmax distributions. Compare constant .5 and the
calibration-target mean, plus paired RMSE differences against the binary control
(500 whole-world bootstrap resamples, seed 0). Apply the existing gates: RMSE <=.10
and beats both constants; negation/conjunction/Bayes errors <=.10; paraphrase and
irrelevant-context MAE <=.05; all Bayes paths defined in raw and calibrated data.
Require four calibration and eight eval worlds and zero errors. Describe intervals
as exploratory: eight worlds, two comparisons, no multiplicity correction.
Passing supports replication; failing remains a result. No tuning on this holdout.

## Implementation and verification plan

1. Create `tests/test_categorical_readout.py`: analytic weighted scores, temperature
   and ordinal bias, invalid/missing logs, prompt has categories but no numerical
   probability request, disjoint fresh worlds, calibration independent of eval
   targets, incomplete/tampered records rejected. Run tests and observe failures.
2. Create `scripts/categorical_readout.py`: validated prompt/scoring, deterministic
   world generation, fixed calibration grid, raw record validation, metrics and
   CLI orchestration. Run new and full CPU tests until passing.
3. Review the implementation and commit protocol/code/tests before inference.
4. Run 504 matched requests with deadline and GPU monitoring; independently
   recompute the saved report. Record failures without post-hoc repair.
5. Write `docs/categorical-readout-2026-09-24.md`, link from README, verify tests,
   commit results, and merge locally to main.
