# Broad seasonal knowledge with constrained output

The structured-output mechanism worked, but this prompt/model setup did not
expose useful seasonal knowledge. All 36 requests returned an allowed word and
stopped normally. The only answers were cold (20) and unknown (16); mild and hot
never appeared. Answer patterns depended strongly on phrasing.

This does not establish that the model lacks weather knowledge internally.
It establishes failure of this particular elicitation and constrained decoding
setup. Before drawing a capability conclusion, an independent Transformers versus
vLLM check and an unconstrained-answer comparison would be useful diagnostics.
Neither was added retrospectively to this pilot.

## Prospective design

The [protocol](plans/2026-09-28-seasonal-knowledge.md) and
[case references](../bench/cases/seasonal-knowledge.json) were frozen before
inference. Six familiar countries were tested in January and July, each with
three variants: neutral scope-first, question-first, and uppercase emphasis on
country/month. Sources and reference labels were evaluator metadata, never model
input. Prompts supplied no climate facts, probabilities, arithmetic, examples,
hemisphere hints or numerical temperature thresholds.

The fixed categorical choices were cold, mild, hot, unknown. We used vLLM 0.11.2
`StructuredOutputsParams(choice=...)`, greedy decoding, seed 0, at most 16 tokens,
no requested logprobs and no calibration. Qwen/Qwen3-0.6B remained pinned to
`c1899de289a04d12100db370d81485cdf75e47ca`, BF16, thinking disabled, with the
existing conservative shared-GPU engine. Structured-output backend selection was
configured as auto. This report does not claim a separately tested backend choice.

Five country-month cases had predeclared qualitative exact labels: January in
Bulgaria, Germany and Canada was cold; both Singapore months were hot. Seven
remaining cases were excluded from exact-label accuracy because country-wide
aggregation and mild/hot boundaries were ambiguous. The labels are broad
analyst interpretations supported by climate references, not measured daytime
threshold classes. National winter/all-day temperature summaries support the
winter interpretation but do not define an exact daytime classification.

References include the [World Bank Bulgaria climate profile](https://climateknowledgeportal.worldbank.org/sites/default/files/2021-06/15848-WB_Bulgaria%20Country%20Profile-WEB.pdf),
[DWD January reference](https://www.dwd.de/EN/press/press_release/EN/2025/20250130_the_weather_in_germany_in_January_2025.pdf?__blob=publicationFile&v=2),
[Canada climate report](https://natural-resources.canada.ca/sites/www.nrcan.gc.ca/files/energy/Climate-change/pdf/CCCR_Chapter4-Temperature%20and%20Precipitation%20Across%20Canada.pdf),
[BOM regional season discussion](https://media.bom.gov.au/social/blog/1989/the-wet-and-the-dry-seasons-in-the-topics/),
[Argentina climate geography](https://www.argentina.gob.ar/node/208806), and
[Meteorological Service Singapore](https://www.weather.gov.sg/climate-climate-of-singapore/).
Detailed rationale and additional seasonal references are versioned with the cases.
Some reference PDFs were accessible through search excerpts only, as recorded in
the protocol. No reference label was changed after observing output.

## Every answer

An asterisk marks an ambiguous case excluded from exact-label accuracy.

| Country | Month | Reference | Neutral | Question first | Emphasis |
| --- | --- | --- | --- | --- | --- |
| Bulgaria | January | cold | cold | unknown | cold |
| Bulgaria | July | Ambiguous* | cold | unknown | cold |
| Germany | January | cold | cold | unknown | cold |
| Germany | July | Ambiguous* | cold | unknown | cold |
| Canada | January | cold | unknown | unknown | cold |
| Canada | July | Ambiguous* | cold | unknown | cold |
| Australia | January | Ambiguous* | unknown | unknown | cold |
| Australia | July | Ambiguous* | cold | unknown | cold |
| Argentina | January | Ambiguous* | unknown | unknown | cold |
| Argentina | July | Ambiguous* | cold | unknown | cold |
| Singapore | January | hot | unknown | unknown | cold |
| Singapore | July | hot | cold | unknown | cold |

## Measurements

- Valid structured outputs: **36/36**, no truncation or request errors.
- Clear-label agreement: **5/15 (33.3%)**. Always answering cold achieves
  **9/15 (60%)** on this prespecified subset.
- Clear cases correct under all three phrasings: **0/5**.
- Cases stable across all three phrasings: **0/12**.
- Pairwise phrasing disagreements: **24/36** within-case comparisons.
- Unknown answers: **16/36 (44.4%)**. Abstention is reported, not interpreted as
  a reliable measure of the model's internal uncertainty.
- Expected directional seasonal contrasts: **0/15**. Seven pairs were tied and
  eight contained unknown; none was a known-label reversal. A tie at this coarse
  resolution is inconclusive rather than a demonstrated climatological error.
- Singapore hot/hot pairs across phrasings: **0/3**.

Question-first yielded unknown on **12/12** cases; emphasis yielded cold on
**12/12**. Neutral yielded cold on eight and unknown on four. That dependence on
phrasing is much clearer than any dependence on country or month. The 36 prompts
represent 12 cases and six country contrasts, not 36 independent knowledge tests.
No numerical prior quality or calibration claim follows from this experiment.

## Verification and provenance

The saved report was recomputed from raw text, stop reasons and manifests, with
an exact match. Exact request text and provenance are checked. All 36 real
prompt-tokenization checks fit within 90 of the 512 available tokens, including
output allowance. CPU tests passed: **125 passed, one optional GPU test skipped**.
The real GPU run independently exercised all 36 constrained generations.

The process exited with status 0 after **36.3 seconds**, including model startup.
Total sampled GPU usage peaked at **4,131 MiB** and returned to **951 MiB** after
exit; no compute worker remained. The runtime emitted cleanup warnings from NCCL
and nanobind after saving the complete report; these are retained in the log.

The initial GPG commit attempt timed out. Before inference, the exact staged
source was frozen as a Git tree and its patch saved alongside the run. The user
then authorized unsigned commits. Implementation commit `6493172` has exactly
the same tree as that prospective snapshot. The manifest's commit/working-tree
fields reflect launch-time state; the snapshot supplies the precise source binding.
No global or repository signing configuration was changed.

Local raw artifacts, ignored by Git under the existing policy:
`runs/seasonal-knowledge-20260928/{manifest.json,records.jsonl,report.json}`,
plus adjacent `.log`, `-gpu.json`, `-source.json` and `-source.patch` files.

Recompute without GPU inference from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/seasonal_knowledge.py \
  --output runs/seasonal-knowledge-20260928 --report-only
```

- Query SHA-256: `d7a26c71fc4023ba03a21805b058f57f6b6c67dbd426fba2555d7aae4e960c23`
- Protocol SHA-256: `22b09dc4ecb78d66a445d80c0f64f22728218331212c51677967d4d4528e712d`
- Prospective source tree: `c2eb0c95b7bf0a9c6bda838156a005cbd7733c14`
- Source patch SHA-256: `1bcdfa9e082a28a1a137df5396c2e5d8040af85fc48e8c13a5740a30defd91e0`
