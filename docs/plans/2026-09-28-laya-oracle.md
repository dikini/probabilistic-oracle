# Laya as a source of Bayesian priors

**Goal:** Evaluate whether and how frozen Laya outputs can supply priors to an
external Bayesian update, separating knowledge, evidence uptake, and probability
readout. Negative results count as evidence; API compatibility alone does not.

**Architecture:** Use Laya's native SDK and English checkpoint directly, in a
separate environment from vLLM. Save complete typed requests, raw distributions,
checkpoint/runtime versions and derived scores. External code performs arithmetic.

**Pinned candidate:** laya 0.3.21; convaiinnovations/laya revision
55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851. Inspect SDK loading and preprocessing,
pin any separately loaded encoder/tokenizer, and check truncation before inference.
Do not auto-route, fine-tune, or fit probabilities on evaluation cases.

## Questions and comparisons, fixed before inference

1. Repeat the 12 weather cases under three phrasings, without and with their
   existing supplied profiles. Use four semantic choices cold/mild/hot/unknown.
   Compare category agreement and stability with Qwen results. The native typed
   interface differs from chat decoding; document the exact mapping.
2. For every country/month, create paired fictional reference passages assigning
   cold versus hot, explicitly overriding real-world knowledge. Test two option
   orders. These 48 requests check whether changed evidence changes the output;
   a correct real-world answer alone cannot demonstrate grounding.
3. Supply finite-population base rates, then ask about an unobserved uniform
   random draw. These probabilities are exact. Use 8 calibration cases with rates
   .1,.2,.3,.4,.6,.7,.8,.9 and 10 test cases with rates .05,.15,.25,.35,.45,
   .55,.65,.75,.85,.95. Calibration uses fictional inventory; test uses fictional
   support records. Separate case identifiers and wording prevent case overlap.
   Each case has four binary choice encodings (A/B label assignment crossed with
   display order), plus native noul. Score choice results separately from noul.
   Preserve raw probabilities. Fit one scalar temperature per readout family on
   calibration only by minimizing expected Bernoulli log loss; report held-out
   raw and calibrated RMSE, expected Brier and log loss, and presentation spread.
4. Propagate those test priors through exact external Bayes with fixed likelihoods
   P(E|H)=.8, P(E|not H)=.2. Compare resulting posterior errors with truth and a
   constant .5 prior. No model likelihood estimation or model arithmetic required.

Weather labels measure category knowledge, not event frequencies. The base-rate
suite measures extraction of supplied priors, not knowledge of real prevalence.
A successful narrow result authorizes only that conditional use. Calibration is
secondary: do not hide failed discrimination with aggregate calibrated scores.

## Implementation and evidence

- Inspect runtime source and freeze request generation and provenance before GPU.
- Add CPU tests for semantic remapping under option permutations, missing or
  malformed distributions, split separation, exact Bayes and scoring baselines.
- Run the pinned English checkpoint locally, with bounded requests and GPU
  monitoring; preserve errors and partial runs rather than filling probabilities.
- Independently recompute reports from raw observations; inspect per-case failure
  patterns and SDK preprocessing. Review code and conclusions before merging.
- Document practical supported uses, unsupported uses, and what a subsequent
  domain training/evaluation round would need. Commit unsigned per user instruction.

Keep all weights, environments and raw run artifacts out of Git. No HTTP service,
retriever, probability calibration claim across domains, or training in this round.
