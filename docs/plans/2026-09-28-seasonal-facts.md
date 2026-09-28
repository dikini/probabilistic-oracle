# Seasonal knowledge with supplied climate facts

**Goal:** Test whether a short, authoritative fact block improves evidence uptake
on the same 36 seasonal questions that failed without supplied facts.

**Architecture:** Extend the seasonal runner with an explicit supplied-facts mode.
Prepend one country-specific block to the unchanged original task text. Save the
facts, their provenance, complete prompt, baseline report/hash and comparison.

## Prospective paired intervention

The user requested supplied knowledge after the no-facts run. This is a diagnostic
follow-up on the same cases, not a fresh holdout or an independent replication.
No calibration, training, arithmetic or new answer categories are introduced.

Use bench/cases/seasonal-facts.json, frozen before this follow-up. The same block
is used for both months and all three phrasings for each country. The block starts
with 'Background facts for this question (take as given):'. It precedes the entire
unchanged original task, so question-first still means question-first within the
task text, following the common fact prefix. Do not change emphasis inside facts.
Each passage supplies explicit month-to-label associations within an evaluator-
defined simplified country profile. Labels: Bulgaria cold/hot; Germany cold/mild;
Canada cold/mild; Australia hot/mild; Argentina hot/cold; Singapore hot/hot, in
January/July order. Ambiguous national labels are task stipulations, not measured
climatological truth. The passage explicitly says it does not describe every region.
Existing climate source URLs are background rationale, not evidence for every
stipulated label. This is manually supplied perfect retrieval, not an actual
retrieval pipeline. The user clarified that this weakly evaluates the RAG direction.

This deliberately makes all 12 cases near-direct extraction tasks.
Improvement demonstrates use of supplied facts under this prompt; it does not
establish learned prior knowledge. Failure leaves instruction following, evidence
uptake and the constrained decoding interface unresolved. The intervention also
changes prompt length and salience. Do not infer a unique causal mechanism.

Retain the same seed, shuffled order, model/revision, engine, thinking disabled,
greedy constrained choices cold/mild/hot/unknown, token budget, deadlines, parsing,
original reference labels, ambiguity exclusions and metrics as the no-facts protocol.
Additionally score agreement against the supplied profile on all 36 prompts,
separately from the unchanged 15-prompt qualitative climate-reference metric.
Compare matched answers, strict-label agreement, unknown counts, stable cases and
seasonal contrasts. Fix facts before inference; no repairs or prompt changes.

## Implementation and verification

1. Add failing tests that supplied facts are included only for their country, the
   old prompt remains an exact suffix, missing fact keys are rejected, and a
   saved baseline yields a matched-answer comparison.
2. Add --facts and --baseline to scripts/seasonal_knowledge.py without changing
   default no-facts behavior. Save the complete fact resource and hashes.
3. Review, run CPU tests and commit unsigned per user instruction before inference.
4. Run the 36 follow-up prompts with bounded GPU monitoring. Recompute raw report,
   compare against original, write all answers and limits, then commit and merge.
