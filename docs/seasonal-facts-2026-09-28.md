# Supplied seasonal knowledge: controlled RAG-generation diagnostic

Supplying explicit answer-bearing passages improved output, but did not establish
reliable grounding. The model agreed with the supplied profiles on **17/36**
prompts. The preceding no-context answers agree with those same subsequently
specified profiles on **7/36** prompts. This latter baseline is a descriptive
paired comparison, not a retroactive change to the first experiment's reference
labels. An always-cold answer agrees with the supplied profiles on **12/36**.

## Intervention

The user requested adding required knowledge to weakly evaluate the RAG direction.
We manually supplied a reference passage for each country: a perfect-retrieval
control for the generation stage. No search index, retriever, ranking, ingestion
or retrieval-quality evaluation was implemented.

The [prospective protocol](plans/2026-09-28-seasonal-facts.md),
[reference profiles](../bench/cases/seasonal-facts.json), runner and tests were
committed unsigned as `848a45c` before inference, with a clean working tree.
Each profile explicitly assigns a daytime temperature category to January and
July. The labels are task stipulations within an evaluator-defined simplified
profile. Especially for large countries, they are not claims of measured,
uniform national temperature or independently validated climatological gold.

For example, the Singapore passage was:

> Reference profile for Singapore. This is a simplified country-level classification for this task, not a description of every region. Typical daytime temperature category in January: hot. Typical daytime temperature category in July: hot.

The passage was prefixed with 'Background facts for this question (take as given):'.
The original task was preserved as an exact suffix, including its unknown/general-
knowledge instruction. Thus this is a context-addition intervention, not a fully
redesigned RAG prompt. That retained instruction may affect source use; no causal
claim uniquely separates knowledge access, evidence salience, instruction
following, and constrained decoding behavior.

Model, revision, BF16 engine settings, no-thinking chat template, greedy decoding,
seed 0, maximum 16 tokens and structured choices were identical to the
[no-context run](seasonal-knowledge-2026-09-28.md). Runtime settings were compared
across saved manifests, allowing only the prompt-template identifier to differ.
Every saved prompt was verified to be exactly its country passage plus the
unchanged baseline prompt. All country/month cases and original ambiguity flags
were unchanged.

## Every follow-up answer

Supplied-profile agreement is evaluated for all cases. Original climate-label
accuracy remains confined to the five clear cases (15 phrasing variants).

| Country | Month | Supplied label | Neutral | Question first | Emphasis |
| --- | --- | --- | --- | --- | --- |
| Bulgaria | January | cold | cold | cold | cold |
| Bulgaria | July | hot | hot | hot | hot |
| Germany | January | cold | cold | cold | cold |
| Germany | July | mild | unknown | mild | cold |
| Canada | January | cold | cold | cold | cold |
| Canada | July | mild | cold | mild | unknown |
| Australia | January | hot | cold | cold | cold |
| Australia | July | mild | unknown | cold | cold |
| Argentina | January | hot | cold | cold | cold |
| Argentina | July | cold | cold | cold | cold |
| Singapore | January | hot | cold | cold | cold |
| Singapore | July | hot | unknown | unknown | cold |

## Paired results

| Measurement | No supplied context | Supplied profile |
| --- | ---: | ---: |
| Valid outputs | 36/36 | 36/36 |
| Agreement with supplied-profile labels | 7/36 | 17/36 |
| Original clear climate-label agreement | 5/15 | 9/15 |
| Cases stable across all phrasings | 0/12 | 8/12 |
| Unknown answers | 16/36 | 5/36 |
| Expected non-tropical seasonal contrasts | 0/15 | 5/15 |
| Singapore hot/hot pairs | 0/3 | 0/3 |

Twenty-one of 36 matched answers changed. Of the eight stable follow-up cases,
five followed the supplied profile and three were stably wrong: Australia January,
Argentina January and Singapore January. The original clear-label score merely
matched its always-cold baseline (9/15); stability and reduced abstention alone
would overstate the improvement.

The model selected cold 26 times, hot three times, mild twice and unknown five
times. All three hot answers were Bulgaria July; the two mild answers were
Germany July and Canada July in question-first form. Northern winter cases and
Argentina July were correct under all phrasings. Both southern January profiles
explicitly said hot, but all answers were cold. Singapore remained wrong under
every phrasing despite its direct hot/hot passage.

The follow-up had seven tied seasonal comparisons and three containing unknown,
with no known-label reversal. None of these pair counts represents an independent
retrieval test: this is the same 12 cases with three phrasing variants.

## Interpretation

The setup can sometimes use supplied context, as shown by Bulgaria July changing
from cold/unknown/cold to hot/hot/hot. It does not reliably extract even directly
stated category labels. This weak RAG-generation diagnostic therefore supports
further interface and grounding checks, not building a larger retrieval system yet.

An independent Transformers-versus-vLLM comparison and constrained-versus-free
answers on the same passages would help distinguish backend/readout behavior from
model instruction following. They were not run as part of this fixed pilot.
No probability calibration, training, prompt repair or post-result tuning occurred.

## Verification and artifacts

All 36 requests completed with allowed outputs and normal stop reasons. The
report was recomputed from raw records with an exact match. The saved baseline
report, its hashes, source profiles and full generated prompts are retained.
CPU verification: **129 tests passed; one optional GPU integration test skipped**.
The longest real prompt including output allowance used 141 of 512 tokens.

The process exited with code 0 after **32.1 seconds**, including startup. Sampled
total GPU memory peaked at **3,992 MiB** and returned to **944 MiB**. Cleanup
warnings from NCCL/nanobind remain in the raw log; no compute worker remained.

Ignored local artifacts:
`runs/seasonal-facts-20260928/{manifest.json,records.jsonl,report.json}`, with
adjacent `.log` and `-gpu.json` files. Recompute without loading the model:

```bash
PYTHONPATH=src .venv/bin/python scripts/seasonal_knowledge.py \
  --output runs/seasonal-facts-20260928 --report-only
```

- Query SHA-256: `a61b2afd1e2dcd7d1d1d504908114b296bb2d6a974f3a22c9d0c3c7eafb394ff`
- Protocol SHA-256: `8b2bba0ff54ebb933158bb97a15d6bb0d2d83ef1d653a9e017eacc8e73b195f2`
- Supplied profile SHA-256: `8fdd928d9812d041e267548731d67d2e7895981f629ba5fff2948302731a25ab`
- Baseline report SHA-256: `54648ccb40a5069266a34816a72b9b7d78a4e3b62514edcae9c3d99fb6f8872c`
