# Probabilistic Oracle

A local experiment testing whether a frozen open-source transformer exposes a
stable, probability-like plausibility signal through its token logits.

The model supplies uncertain judgments. External code computes probability
identities and Bayesian updates. A normalized token score is a plausibility
score; interpreting it as a probability requires empirical evidence.

## First milestone

- **oracle:** one vLLM backend using a Hugging Face model, proposition scoring,
  and configurable binary verbalizers.
- **algebra:** negation, conjunction through conditional decomposition, Bayesian
  updates, and discrepancy metrics.
- **bench:** reproducible synthetic cases with known probabilities and a small
  collection of ordinary-world propositions.

Measure calibration, paraphrase variance, negation consistency, agreement between
direct and algebraic Bayesian updates, and sensitivity to irrelevant context.

Defer memory/RAG, semantic extraction, hypothesis generation, belief graphs,
coherence projection, decision procedures, hidden-state probes, and fine-tuning.

## Status

Repository initialized with the agreed scope and a proposed implementation plan.
There is no executable implementation or experimental result yet.

- [Milestone design](docs/plans/2026-09-24-milestone-1-design.md)
- [Implementation plan](docs/plans/2026-09-24-milestone-1.md)

Development is local and tracked in Git. Model weights, caches, credentials, and
run outputs stay outside version control. Model, hardware, dependency versions,
and numerical acceptance thresholds remain to be selected.
