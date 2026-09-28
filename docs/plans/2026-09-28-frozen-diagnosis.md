# Frozen diagnosis: direct, factorized and hybrid inference

**Goal:** Compare frozen Laya final answers with Laya-supplied probabilistic
components and evidence extraction followed by exact external Bayes. Separate
model error, evidence-extraction error, and incorrect independence assumptions.
No neural training, calibration fitting, or Bayesian parameter fitting this round.

**Runtime:** Reuse pinned Laya 0.3.21 and checkpoint
55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 in the isolated Laya environment. Capture
raw logits and shipped probabilities with the verified SDK recording interface.
Use shipped scaling as the primary readout, unscaled logits as a secondary check.
Keep decoder/model fixed, reject input truncation, log actual device and runtime.

## Finite diagnosis worlds

Exactly one fault: coolant, bearing, or sensor, with priors .5/.3/.2. Two symptoms:
high temperature (X) and vibration (Y). P(X|fault)=.8/.4/.1;
P(Y|fault)=.7/.3/.2. All numbers are stipulated fictional population frequencies.
No real machinery expertise is required or implied.

Independent world: each joint is the product of its two conditional marginals.
Correlated world: P(X=1,Y=1|fault)=.68/.28/.09; derive the other joint cells while
preserving both marginals. Both worlds have strictly positive cells. Fully enumerate
the four observable symptom patterns; do not invent sampled diagnosis labels.
Compute exact posterior targets and probability-of-pattern weights externally.

## Information and presentation conditions

Full: provide priors, both symptom marginals, and all four joint cells per fault.
Reduced: omit joint cells; provide the same priors and marginals. Reduced prompts
must be identical across worlds. Never disclose the world identifier or correlation
regime in a model input. Removing joint information makes true dependencies
unidentifiable; do not characterize unavoidable information loss as model failure.

Three fixed paraphrases of each question and symptom report, crossed with normal
and reversed option order. Enumerate 2 worlds x 2 coverage conditions x 4 patterns
x 3 phrasings x 2 orders = 96 diagnosis presentations. This is eight underlying
world/pattern cases, not 96 independent observations.

## Approaches and ablations

- Direct: one native three-choice question returns a final fault distribution from
  table plus report.
- Factorized joint (full only): one prior question plus three four-way joint
  symptom-distribution questions, one per fault. External Bayes uses these and
  the extracted symptom pattern. Prior/likelihood prompts omit the case report,
  preventing posterior leakage into parameters.
- Factorized naive (both coverage levels): one prior plus six binary conditional
  symptom questions. External code multiplies marginal likelihoods, then updates.
- Hybrid joint (full only): Laya extracts one of four symptom states from the
  report; external code applies the provided exact prior and joint table.
- Hybrid naive (both): same extraction, external provided prior and marginals.

Share the same evidence-extraction observation between factorized and hybrid
methods. Reuse parameter observations across the four reports in their world,
coverage, phrasing and order condition. Also report factorized calculations using
true evidence, to separate parameter errors from extraction errors.

Controls: true-evidence exact joint (hidden-truth reference, not a reduced-information
competitor), true-evidence naive (isolates structural error), and prior-only.
Joint methods are unavailable when the joint table is removed; do not silently
supply them hidden reference parameters. Reduced direct and naive methods remain
comparable on exactly the provided information.

## Metrics and cost

Primary: expected classification accuracy, expected multiclass log loss and Brier,
weighted by the exact observation distribution within each world. Average six
presentations equally. Also report per-case posteriors, maximum total-variation
change across phrasings/orders, extraction accuracy, and parameter errors.
Log loss uses an explicit 1e-12 numerical floor, with zero counts reported.
Count requests and sum measured forward-call seconds by method, distinguishing
shared, one-time parameter estimation from per-report cost. Record end-to-end
process wall time and sampled GPU memory. Do not treat reused parameters as free.

## Implementation and completion evidence

1. CPU-test normalized worlds, equal marginals, exact Bayes, an independent-world
   equality control, correlated-world divergence, omission and prompt equality,
   semantic option remapping, scoring and missing-record rejection.
2. Freeze requests and source in unsigned commits before model inference.
3. Execute bounded offline GPU inference (ten-minute watchdog), preserving complete
   prompts, token IDs, logits, SDK responses, hashes, settings and failures.
4. Recompute and independently review reports. Diagnose unexpected measurement
   behavior before conclusions. Document how table removal changes error and the
   relative performance of all three approaches; commit and merge locally.

No marginal-likelihood or Occam comparison is planned: first establish predictive
behavior. Known parameters are privileged correctness controls for extraction and
structure, not evidence that a learned Bayesian model would have the same quality.
