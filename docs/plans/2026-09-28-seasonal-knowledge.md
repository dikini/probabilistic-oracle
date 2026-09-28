# Broad seasonal knowledge with constrained categorical output

**Goal:** Test whether the frozen small model exposes broad seasonal knowledge,
without arithmetic, numerical probabilities, supplied climate facts or calibration.

**Architecture:** A standalone runner uses the existing pinned vLLM engine and
StructuredOutputsParams(choice=...). A versioned case file fixes reference labels,
ambiguity and seasonal contrasts before inference. Raw text and finish reasons
are saved; reporting checks completeness and compares phrasing variants.

**Tech stack:** Qwen3-0.6B, vLLM 0.11.2, Python standard library, pytest.

## Prospective protocol

Six countries: Bulgaria, Germany, Canada, Australia, Argentina, Singapore.
Two months: January and July. Three phrasings: neutral scope-first, question-first,
and country/month uppercase emphasis. Total 36 requests, deterministic shuffled
order with seed 2026092801. No prompt changes or reference revisions after results.

Ask which category best describes typical daytime temperatures, considering the
country as a whole and allowing regional variation. Choices are cold, mild, hot,
unknown. Unknown is permitted when the model cannot select from general knowledge;
it is not assumed to accurately diagnose uncertainty. Keep choice order fixed.
No degrees, thresholds, examples, hemisphere hints, reference labels, or source
material are supplied. Only task and answer-format instructions change position;
uppercase emphasis changes no factual content. Prompt variants and schema are
versioned and hashed in the manifest.

Use the pinned model/revision, thinking disabled, greedy decoding, seed 0,
max_tokens=16, constrained choice output, no requested logprobs and no calibration.
Retain settings, structured-output engine config, prompts/token IDs, raw output,
generated tokens, finish reason, runtime, code revision and manifest hash.
A valid answer must strip to one permitted word and finish with reason stop.
Never repair answers or silently accept truncation. Incomplete runs receive no
knowledge-success verdict. External five-minute process limit; cooperative
four-minute inference deadline. Model startup is included in those limits.

## References fixed before inference

Labels are analyst-defined qualitative interpretations supported by climate
sources, not temperature thresholds or empirically fitted gold classes. The
terms mild/hot and country-wide aggregation are ambiguous; no land-area or
population weighting is specified in this deliberately broad prompt.

Strict exact-label subset (5 cases, 15 prompts): Bulgaria January=cold, Germany
January=cold, Canada January=cold, Singapore January=hot and July=hot. January
national/all-day means support the winter label but are not daytime measurements;
this is a broad everyday-language judgment, not a climatological calculation.
All seven remaining cases are ambiguous and excluded from exact-label accuracy.
Report their raw answers without retrospectively accepting or rejecting labels.

Expected ordinal direction: July warmer than January for Bulgaria/Germany/Canada;
January warmer than July for Australia/Argentina. For Singapore expect hot in both
months, not merely any unchanged word. Known labels ordered cold<mild<hot.
For non-tropical countries report expected/reversed/tied/unknown separately;
a tie is inconclusive at this coarse resolution, not a demonstrated reversal.
No majority vote: report each phrasing and country pair independently.

Climate sources and rationale are in bench/cases/seasonal-knowledge.json. Reference
URLs are evaluator metadata only. World Bank Bulgaria and DWD January summaries
were available through search excerpts; direct full-text fetches were blocked.
Other references include Canada's climate report, Australian Bureau of Meteorology,
Argentina government geography/seasonal report, and Meteorological Service Singapore.

## Reporting

Report valid coverage, answer counts, unknown count, exact agreement on 15 strict
prompts and number of 5 strict cases correct in all variants; report a constant-cold
baseline (9/15). Report all three answers for each case, stability across all three
variants, pairwise disagreement count, seasonal directions per country/variant,
and Singapore hot/hot consistency. Preserve every error in an incomplete report.
There are 12 cases, not 36 independent knowledge examples. No confidence interval,
probability calibration, broad model-capability claim, or previous benchmark gate.
This combines a new domain, semantic answer words and constrained decoding; a
positive result cannot isolate which change helps relative to older experiments.

## Implementation plan

1. Version cases, references and this protocol. Add tests for prompt variants,
   strict parsing, incomplete/duplicate rejection, stable-but-wrong answers,
   ambiguity exclusion, seasonal reversal/ties, and unknown handling.
2. Implement scripts/seasonal_knowledge.py using the existing backend, with
   manifest/raw-record validation and CPU report-only mode. Pass full tests.
3. Review and commit protocol/cases/runner/tests before inference. Run all 36
   actual GPU requests with structured output and bounded monitoring.
4. Recompute saved report, document all answers and limits, link README, verify,
   commit and merge locally to main. Do not run additional tuned experiments.

## Prospective execution exception

Before inference, the configured GPG signer timed out. To avoid blocking the
requested read-only experiment, freeze the staged source as a Git tree hash and
save the staged patch plus its SHA-256 adjacent to the run before launching the
model. The manifest records the preexisting commit and non-clean working tree;
the adjacent source snapshot identifies the exact prospective implementation.
Signed commit creation remains pending user action; no signing setting is changed.
This replaces the pre-inference commit step for this run only. No labels, prompts
or scoring rules change.
