# Milestone 1: test the plausibility signal

## Status and source

The scope follows the user-approved first milestone in the conversation
[Replicate Jev Probabilities](chatgpt-conversation://6aafd354-edf4-83ed-a3ea-bfd353e68703).
This repository starts a separate project; it does not change Ash.
Implementation details below are proposals, not established experimental facts.

## Question

Does a frozen open-source transformer expose a sufficiently stable and
probability-like plausibility signal through its logits to justify further work?

## Agreed boundary

Use a Hugging Face model with vLLM. Query explicit propositions in explicit
contexts. Compare independent model judgments with exact external probability
operations. Retain oracle, algebra, and bench only in this milestone.

Memory, semantic extraction, hypothesis generation, general symbolic reasoning,
coherence projection, hidden-state probes, and weight updates are deferred.

## Proposed scoring contract

A score request contains context, proposition, and a prompt/verbalizer choice.
A result retains the exact prompt, candidate token IDs, raw token log-probabilities,
normalized binary score, model/tokenizer revisions, and inference settings.
Initially require each verbalizer to be one token at the actual scoring boundary.
Reject unsupported or ambiguous tokenization rather than approximating it.

Normalize the two candidate log-probabilities using a stable two-way softmax.
Retain their original values so normalization does not hide low total candidate
mass. Verify extraction before any sampling truncation or vocabulary constraint
that could change the measured distribution. Missing candidate probabilities are
explicit failures. Calibration is a separate transformation fitted on separate
cases, with its parameters and provenance retained.

## Algebra and experiments

Query H and its negation independently to measure complement discrepancy.
For Bayesian consistency, query H, E assuming H, E assuming not-H, and H after
observing E. Compare the last score with the Bayesian update from the first three.
Keep assumptions distinct from observations in prompt templates.

Validate finite inputs in [0, 1]. A zero-probability conditioning event produces
an explicit undefined result, not a fabricated posterior. Query conjunctions
independently and compare with products using conditional scores; do not assume
independence.

Synthetic cases provide known targets. Ordinary-world cases support stability
checks; calibration claims require defensible labels or reference probabilities.
Separate calibration fitting and evaluation by underlying case, keeping related
paraphrases in the same split. Report raw and calibrated performance separately.
Include simple baselines and uncertainty across cases. Consistency alone does not
establish accuracy or calibration.

## Evidence and retention

Save reproducible JSON runs first; add Parquet if analysis requires it. Record
case IDs, splits, prompts, responses, errors, settings, and code revision.
GPU-free tests cover algebra, parsing, metrics, and run records. A separate real
model smoke run verifies tokenization and scoring behavior.

The deliverable is a reproducible report, including negative findings. Choose
quantitative acceptance criteria before evaluating the held-out cases. Continue
to memory and semantic extraction only if measurements justify that investment.

## Open implementation choices

Check available GPU memory and select one compatible model and pinned inference
stack. Verify the backend API against that version's primary documentation.
Choose prompt variants, labeled cases, calibration method, and acceptance
thresholds before the main experiment.
