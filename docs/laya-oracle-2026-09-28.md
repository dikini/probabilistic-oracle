# Laya as an oracle for Bayesian priors

The tested frozen English Laya checkpoint is useful as a candidate evidence
classifier, but these experiments do not support treating its answer probabilities
as reliable Bayesian priors. It followed all 48 direct changed-fact controls and
classified 119/120 presentations of synthetic support records correctly. Direct
probabilities for explicitly specified random draws were much less reliable.

The practical route supported by this pilot is **documents → Laya categories →
external prevalence estimation → a prior for a later case**. This separates the
model's semantic decision from the statistical meaning of an event frequency.
It requires representative source records and measured classification error before
real use. We have demonstrated the mechanics on synthetic examples, not established
real-world priors or a deployable classifier.

## Model and measurement

The [prospective protocol](plans/2026-09-28-laya-oracle.md) was committed as
`9bd698d`. Initial inference used clean `4358c24`; the exploratory follow-up used
clean `4efb437`. Both are unsigned, as requested. The original protocol was fixed
before the initial results. The follow-up was deliberately specified after them.

- Model: `convaiinnovations/laya`, English root checkpoint, revision
  `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.
- Runtime: Laya 0.3.21, PyTorch 2.9.0, Transformers 4.57.6; full environment in
  [requirements-laya.lock](../requirements-laya.lock).
- Direct SDK, CUDA, BF16 autocast, no quantization, no compilation or TileLang,
  no routing or model training. Checkpoint weight file: 842,609,210 bytes.
- Encoder config and tokenizer are bundled in the same pinned snapshot. Inference
  ran with `HF_HUB_OFFLINE=1`; no unpinned fallback was fetched. SHA-256 hashes
  of all five snapshot files are saved with the initial run.
- The SDK applies shipped temperatures and rounds probabilities to four decimals.
  We also captured the exact float-converted decision logits from the same forward
  pass. Raw means softmax at temperature 1; shipped means checkpoint scaling;
  fitted means a separate scalar fit using only our calibration cases.
- Every request records state, typed question, semantic mapping, token IDs, option
  markers, raw logits, full SDK response, effective temperature, device and timing.
  The runner compares bounded versus expanded tokenization and rejects truncation.

All 402 responses were valid, stayed on CUDA, and used distinct option token spans.
Captured vector length, reconstructed shipped probabilities and choice argmax were
verified against SDK outputs, allowing only four-decimal rounding. Saved token
counts also agree with SDK usage. Both reports recompute exactly from raw records.
The SDK consistency guard was added after the first run and applied retrospectively
without altering its observations.

## Knowledge and supplied evidence

The initial layout preserved each original chat task as the state and asked Laya
to answer its temperature question. The follow-up used Laya's native layout:
original task wording in question instructions; supplied passage, or the neutral
'Use general world knowledge.' cue, in state. Thus these compare interfaces as
well as model families; they are not byte-identical model inputs across backends.

| Measurement | Qwen 0.6B | Qwen 4B AWQ | Laya initial | Laya native follow-up |
| --- | ---: | ---: | ---: | ---: |
| No facts: clear climate labels | 5/15 | 15/15 | 9/15 | 1/15 |
| Supplied facts: stipulated profile labels | 17/36 | 36/36 | 30/36 | 35/36 |
| Supplied facts: clear climate labels | 9/15 | 15/15 | 15/15 | 15/15 |
| No facts: cases stable across phrasing | 0/12 | 10/12 | 12/12 | 11/12 |
| Supplied facts: cases stable across phrasing | 8/12 | 12/12 | 8/12 | 11/12 |

Laya initially answered cold on every no-facts prompt, exactly matching the
always-cold baseline on clear labels. Native layout mostly changed that collapse
to mild: only neutral Canada January was cold. Neither result demonstrates useful
unaided climate knowledge. It also does not prove the encoder contains no climate
information; we have tested two interfaces to this decision head.

Supplied profiles substantially improved category extraction. They remain
simplified task stipulations, not empirical national climatology. The native
follow-up missed Canada July under emphasis, choosing cold instead of mild.
The 15 clear-label prompts represent five underlying facts, not 15 independent
knowledge samples. All cases were reused from earlier diagnostics.

In 48 changed-fact controls, every country/month was explicitly assigned cold or
hot in a fictional profile, with both category orders tested. Laya followed the
assigned category in **48/48**, including assignments contrary to real climate.
Mean shipped probability of the assigned category was 0.931. This establishes
sensitivity to this direct supplied evidence, not calibration or robust RAG.

## Direct priors from stated base rates

Eight calibration populations used packages with amber/violet marks. Ten held-out
populations used billing/delivery support records, with true billing probabilities
.05, .15, ..., .95. Each population contained 100 items, stated both class counts,
and asked about a uniform random draw whose identity was unobserved. No sampled
outcome was supplied. Hence the target is the exact draw probability, not the
chance that an observed document's classification is correct.

Four binary choice encodings crossed A/B assignment with display order; `noul`
was tested separately. All encodings of a case remained in one split. Temperature
fitting used calibration populations only, minimizing expected Bernoulli log loss
over a prespecified implementation grid from 0.1 to 100. No evaluation-case fitting.

| Readout | Prior RMSE raw | Prior RMSE shipped | Prior RMSE fitted | Fitted temperature |
| --- | ---: | ---: | ---: | ---: |
| choice | 0.2855 | 0.2713 | 0.2736 | 1.496 |
| noul | 0.5387 | 0.4341 | 0.2863 | 100.000 |
| Constant 0.5 | 0.2872 | 0.2872 | 0.2872 | — |

Choice gives only a small aggregate improvement over the constant. Raw expected
log loss is 0.6946 versus 0.6931 for constant 0.5; shipped log loss is 0.6749.
Raw expected Brier is 0.2490 versus 0.2500 for the constant. These are expectations
over the known Bernoulli event, not scores against invented sampled outcomes.
Presentation effects remain large: the maximum within-case raw probability range
is **0.5694**. These 40 choice observations represent ten distinct rate cases.

Native `noul` assigns raw positive probabilities from 0.915 to 0.990 across true
rates from 0.05 to 0.95. The fitted temperature reaches the grid ceiling of 100,
flattening predictions near 0.5. This is essentially retreat to the constant,
not evidence that calibration repaired discrimination.

For each true rate, the following ranges cover the four choice presentations;
the native binary result uses its shipped scaling.

| True prior | Choice raw range | Choice shipped range | Noul shipped |
| --- | --- | --- | ---: |
| 0.05 | 0.144–0.453 | 0.282–0.475 | 0.768 |
| 0.15 | 0.150–0.506 | 0.287–0.503 | 0.824 |
| 0.25 | 0.213–0.571 | 0.335–0.537 | 0.844 |
| 0.35 | 0.199–0.509 | 0.325–0.505 | 0.851 |
| 0.45 | 0.164–0.530 | 0.299–0.516 | 0.817 |
| 0.55 | 0.270–0.471 | 0.372–0.485 | 0.850 |
| 0.65 | 0.280–0.571 | 0.379–0.537 | 0.909 |
| 0.75 | 0.292–0.559 | 0.386–0.531 | 0.865 |
| 0.85 | 0.276–0.627 | 0.376–0.568 | 0.878 |
| 0.95 | 0.208–0.777 | 0.331–0.658 | 0.869 |

External Bayes used fixed P(E|H)=0.8 and P(E|not H)=0.2; Laya did not estimate
likelihoods or perform the update. Posterior RMSE for choice was 0.2405 raw,
0.2371 shipped and 0.2345 fitted, versus 0.2628 from a constant 0.5 prior.
For noul it was 0.3664, 0.3378 and 0.2639, respectively. These modest aggregate
choice gains do not establish a dependable prior, given the large label effects
and small designed test set. A different likelihood ratio could change the errors.

A classification head trained to recognize a class from observed text need not
represent the aleatoric uncertainty of a hidden random draw. This task mismatch
is one plausible explanation. This experiment cannot isolate training objective,
numeric comprehension, semantic discrimination and label bias as unique causes.
When source counts are already explicit, deterministic parsing and arithmetic
are preferable to asking this head to reconstruct a probability distribution.

## Deriving priors by counting classified records

The exploratory corpus contained 60 fictional support records in three cohorts
of 20, constructed with 4, 10 and 16 billing cases. It reused up to 20 short phrase
templates; these are easy synthetic cases, not 60 independently authored examples.
Each was classified using descriptive billing/delivery keys in both display orders.
No numeric prevalence was given to Laya. Each order is analyzed separately:
120 presentations are not counted as 120 distinct population observations.

Laya was correct on 119/120 presentations, with one order disagreement. In reversed
order, it sent 'Please leave the shipment at the collection point.' to billing
with shipped probability 0.5782. The other order correctly selected delivery.

| Cohort | Known billing fraction | Estimated, billing first | Estimated, delivery first |
| --- | ---: | ---: | ---: |
| Low | 4/20 = 0.20 | 0.20 | 0.20 |
| Balanced | 10/20 = 0.50 | 0.50 | 0.55 |
| High | 16/20 = 0.80 | 0.80 | 0.80 |

An illustrative uniform Beta(1,1) starting distribution gives

    prevalence | classified records ~ Beta(1 + billing_count, 1 + other_count)
    predictive P(next record is billing) = (1 + billing_count) / (2 + record_count)

For billing-first order, external predictive probabilities are 5/22 = 0.2273,
11/22 = 0.5 and 17/22 = 0.7727. Reversing order changes only the balanced cohort
to 12/22 = 0.5455. These equal the known-label calculation except for the one
misclassification. The Beta distribution is a posterior over prevalence; its
predictive probability can serve as a prior for a later case before case-specific
evidence. The uniform starting distribution was chosen by us, not learned by Laya.

This calculation assumes exchangeable representative records and treats labels
as correct. Neither assumption is established for real data here. A deployment
needs a labeled audit set for false positives/negatives, sampling provenance,
drift checks and a model for classification error. For multiple exclusive classes,
external Dirichlet counts are a natural extension. For dependent attributes,
estimate conditional or joint counts; do not multiply marginal model scores and
assume independence. Those extensions were not tested in this round.

## Conclusion and next experiment

Use Laya provisionally as a compact semantic extractor in front of explicit
statistical machinery. Use `choice` with meaningful category descriptions and
test order changes. Do not use its confidence, entropy-based `confidence` field,
or `noul` directly as a prior without domain evidence. Do not mistake successful
extraction from a passage for independent knowledge of world prevalence.

The next useful validation is a fresh, realistically sampled, manually labeled
record set from one narrow domain. Keep a held-out set for classification error
and prevalence estimation, compare with simple baselines, and evaluate sensitivity
to sampling and prompt changes. Fine-tuning may improve that classifier, but this
round provides no measured training result or guarantee of calibrated priors.
Qwen3-4B-AWQ remains the stronger unaided weather model in our measured setups.

## Artifacts and reproduction

Initial run: `runs/laya-oracle-20260928/`, 210 requests in 23.8 seconds including
startup, sampled peak total GPU memory 3,721 MiB. Follow-up:
`runs/laya-oracle-followup-20260928/`, 192 requests in 16.6 seconds, peak 3,694 MiB.
Both processes exited 0; GPU memory returned to GUI baseline. No model remains
loaded. No need for quantization was observed on the shared RTX 4050 laptop.

Each directory holds `manifest.json`, `records.jsonl`, `report.json`. Adjacent
`.log` and `-gpu.json` files preserve execution and sampled memory. The initial
run also has `checkpoint-sha256.json`; an adjacent environment file is retained.
Raw runs and model weights are ignored by Git. CPU verification: **139 passed,
one optional vLLM GPU integration test skipped**; actual Laya runs are separate.

```bash
uv venv .venv-laya --python 3.12
uv pip sync --python .venv-laya/bin/python requirements-laya.lock
USE_TF=0 .venv-laya/bin/python scripts/laya_oracle.py \
  --output runs/new-laya-initial --deadline "$RUN_DEADLINE"
USE_TF=0 .venv-laya/bin/python scripts/laya_oracle.py --followup \
  --output runs/new-laya-followup --deadline "$RUN_DEADLINE"
```

Use an ISO deadline in the future and new directories. The first load may download
the pinned snapshot; subsequent runs can set `HF_HUB_OFFLINE=1`. The original runs
also had a ten-minute process-group watchdog, never reached. Recompute a saved
report without model loading using `--output DIRECTORY --report-only`.

Request SHA-256 hashes:

- Initial: `38108d2381912f889b17ef5b517b49e7cabca8cff59b20833bd1fcdde632a816`
- Follow-up: `6d66606ad8ea05ce0777a2be9e23b7d00bec4893388357cd1134f35785e17df4`

## All weather answers

Triplets are neutral / question first / emphasis. Original no-facts answers are
cold/cold/cold for every case. Native no-facts answers are mild/mild/mild except
Canada January cold/mild/mild.

| Case | Supplied label | Initial supplied facts | Native supplied facts |
| --- | --- | --- | --- |
| bulgaria-january | cold | cold / cold / cold | cold / cold / cold |
| bulgaria-july | hot | hot / hot / hot | hot / hot / hot |
| germany-january | cold | cold / cold / cold | cold / cold / cold |
| germany-july | mild | mild / cold / mild | mild / mild / mild |
| canada-january | cold | cold / cold / cold | cold / cold / cold |
| canada-july | mild | mild / cold / unknown | mild / mild / cold |
| australia-january | hot | hot / hot / hot | hot / hot / hot |
| australia-july | mild | mild / hot / hot | mild / mild / mild |
| argentina-january | hot | hot / hot / hot | hot / hot / hot |
| argentina-july | cold | cold / cold / unknown | cold / cold / cold |
| singapore-january | hot | hot / hot / hot | hot / hot / hot |
| singapore-july | hot | hot / hot / hot | hot / hot / hot |
