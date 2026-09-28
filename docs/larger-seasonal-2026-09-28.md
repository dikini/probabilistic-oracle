# Larger quantized model: matched seasonal tests

Qwen3-4B-AWQ substantially improved both unaided broad knowledge and use of
supplied passages on this small, reused diagnostic set. Without supplied facts,
it matched all 15 clear climate labels, compared with 5/15 for Qwen3-0.6B.
With supplied facts, it followed every stipulated label under all three phrasings:
36/36, compared with 17/36. This supports continuing with the larger local model
for knowledge and grounding experiments, but does not establish calibrated priors.

## Controlled comparison

The [prospective protocol](plans/2026-09-28-larger-seasonal.md) and model-selection
code were committed unsigned as `b87125f` before inference. Both runs used that
clean source revision. The official [Qwen3-4B-AWQ checkpoint](https://huggingface.co/Qwen/Qwen3-4B-AWQ)
was pinned to `74d4bd2bd4bff9cafc9345221320bffb08b406a3` for both model and tokenizer.
Weights occupy 2,666,027,672 bytes; quantization is 4-bit AWQ, group size 128.
vLLM 0.11.2 used AWQ Marlin with FP16 activations, a 65% GPU memory budget,
one sequence, 512-token context and eager execution. No resource retry was needed.

The 0.6B reference used BF16 and a 50% memory budget. Thus size, quantization,
activation dtype and inference kernel change together. We cannot attribute gains
to model size alone or estimate quality lost to quantization.

All 72 request texts, rendered chat prompts and token-ID sequences were verified
identical to their corresponding earlier 0.6B inputs. Greedy decoding, seed 0,
maximum 16 tokens, disabled thinking, structured choices, shuffled order,
reference labels and metrics were unchanged. This repeats the latest weather
experiments, not the earlier card/probability or token-logit benchmarks.

## Results

| Measurement | 0.6B no facts | 4B AWQ no facts | 0.6B supplied facts | 4B AWQ supplied facts |
| --- | ---: | ---: | ---: | ---: |
| Valid outputs | 36/36 | 36/36 | 36/36 | 36/36 |
| Clear climate-label agreement | 5/15 | 15/15 | 9/15 | 15/15 |
| Agreement with supplied-profile labels | 7/36 | 25/36 | 17/36 | 36/36 |
| Cases stable across three phrasings | 0/12 | 10/12 | 8/12 | 12/12 |
| Unknown answers | 16/36 | 3/36 | 5/36 | 0/36 |
| Expected seasonal contrasts | 0/15 | 7/15 | 5/15 | 15/15 |
| Singapore hot/hot pairs | 0/3 | 3/3 | 0/3 | 3/3 |

The clear-label metric contains five cases, each repeated under three phrasings;
these are not 15 independent facts. The supplied labels are simplified task
stipulations, not measured national climatological truth. No-facts agreement with
those labels is a descriptive comparison, not an expansion of the original gold
set. Always-cold baselines are 9/15 for clear labels and 12/36 for supplied labels.

Unaided 4B answers still show substantive gaps. Australia is hot in both months;
Argentina has reversed seasons under neutral and question-first phrasing, with
January unknown under emphasis. Canada July alternates mild and unknown. The
unaided seasonal comparison has seven expected, three tied, three unknown and
two reversed outcomes. Stability alone therefore remains insufficient.

Adding the passages changes 11 answers, all into agreement with the supplied
profile, and removes the remaining phrasing disagreement. This is encouraging
for using retrieved context: the generation stage can use these direct passages
with this model and prompt. It does not evaluate actual retrieval, conflicting or
irrelevant passages, multi-document synthesis, or fresh cases. Both the larger
model's learned knowledge and its ability to follow explicit context appear more
useful here; their internal mechanisms are not identified by this experiment.

## Every larger-model answer

Each triplet is neutral / question first / emphasis.

| Country | Month | Supplied label | No facts | With facts |
| --- | --- | --- | --- | --- |
| Bulgaria | January | cold | cold / cold / cold | cold / cold / cold |
| Bulgaria | July | hot | hot / hot / hot | hot / hot / hot |
| Germany | January | cold | cold / cold / cold | cold / cold / cold |
| Germany | July | mild | mild / mild / mild | mild / mild / mild |
| Canada | January | cold | cold / cold / cold | cold / cold / cold |
| Canada | July | mild | unknown / mild / unknown | mild / mild / mild |
| Australia | January | hot | hot / hot / hot | hot / hot / hot |
| Australia | July | mild | hot / hot / hot | mild / mild / mild |
| Argentina | January | hot | cold / cold / unknown | hot / hot / hot |
| Argentina | July | cold | hot / hot / hot | cold / cold / cold |
| Singapore | January | hot | hot / hot / hot | hot / hot / hot |
| Singapore | July | hot | hot / hot / hot | hot / hot / hot |

## Verification, runtime and reproduction

Both 36-request processes exited normally: 37.4 seconds without facts and 36.2
seconds with facts, including startup. The pinned model download took about 80
seconds beforehand. Sampled total GPU memory peaked at 5,042 MiB and 5,020 MiB,
respectively, with minimum reported free memory 721 MiB and 743 MiB. After both
runs, GPU use returned to 1,088 MiB. The GUI remained resident; responsiveness
was not formally measured. Existing NCCL/nanobind shutdown warnings remain in logs.

Both reports reproduced exactly from raw saved records. The facts baseline was
independently verified to use the same 4B engine and sampling. Following review,
the runner now rejects a baseline from a different model/engine or sampling setup
and retains its backend provenance. That guard was added after these runs and
does not alter their prompts or observations. CPU verification: 133 passed, one
optional GPU integration test skipped. These 72 actual GPU requests are separate
from that optional test. Longest prompt plus allowance: 90 tokens without facts,
141 with facts, within 512.

Ignored artifacts are in `runs/seasonal-knowledge-qwen3-4b-awq-20260928/` and
`runs/seasonal-facts-qwen3-4b-awq-20260928/`, each containing manifest, raw records
and report, with adjacent `.log` and `-gpu.json` files. Model weights remain in
the Hugging Face cache, outside Git. Original runs remain intact.

To reproduce, choose new output directories and an ISO deadline in the future:

```bash
PYTHONPATH=src .venv/bin/python scripts/seasonal_knowledge.py \
  --model-profile qwen3-4b-awq --output runs/new-4b-no-facts \
  --deadline "$RUN_DEADLINE"
PYTHONPATH=src .venv/bin/python scripts/seasonal_knowledge.py \
  --model-profile qwen3-4b-awq --facts --baseline runs/new-4b-no-facts \
  --output runs/new-4b-facts --deadline "$RUN_DEADLINE"
```

Recompute either saved report without loading a model using `--output DIRECTORY
--report-only`. Default model selection remains Qwen3-0.6B.

| Artifact hash | No facts | Supplied facts |
| --- | --- | --- |
| query_sha256 | `d7a26c71fc4023ba03a21805b058f57f6b6c67dbd426fba2555d7aae4e960c23` | `a61b2afd1e2dcd7d1d1d504908114b296bb2d6a974f3a22c9d0c3c7eafb394ff` |
| protocol_sha256 | `311e219be54e96e51f8877f7cb2d2144c1a4be62cfe2680812f2311c2b5bf672` | `b40a0a87745ffb27bbaea452c77622a6f8a3a625a2e18bb7d1dd0f0aa80e93c0` |
