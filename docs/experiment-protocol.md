# First benchmark protocol

Status: prospective exploratory protocol, fixed before the first benchmark run.
The four earlier hardware-probe prompts were development checks, not this
held-out evaluation. This small synthetic suite is not a population sample.

## Cases and split

Use the versioned `bench/cases/synthetic.json`: four calibration worlds and eight
evaluation worlds. Each world is a finite deck of uniformly selected cards with
an exhaustive red/blue and circle/no-circle contingency table. Exact rational
counts define all six targets: H, not-H, E given H, E given not-H, H given E,
and H-and-E. Assumptions and observations have separate prompt sections.

Each world has base, paraphrased-context, and irrelevant-context variants.
All queries and variants from one world stay in the same split. This produces
216 synthetic requests. Four ordinary-world cases add 24 stability-only
requests; they supply no calibration targets. Total: 240 requests per verbalizer
pair. Default TRUE/FALSE; YES/NO and A/B are optional separate runs, each checked
for single-token boundaries. The first experiment uses only TRUE/FALSE.

## Raw scores and calibration

The affirmative score is the stable two-way softmax of raw token log-probabilities.
Retain both raw logs and their total vocabulary probability mass. The vLLM
0.11.2 adapter returns the full vocabulary, then extracts the candidates.
Use the already verified Qwen3-0.6B revision and conservative laptop settings.

Fit a single temperature on the 24 base queries from calibration worlds only.
Choose the temperature minimizing soft-target binary log loss from the fixed
grid [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 8]. Clip scores to [1e-12, 1-1e-12]
only for logarithms and temperature scaling; retain originals unchanged.
Ties favor the temperature closest to 1. The transformation rescales binary
log-odds; it does not change model weights or inference sampling temperature.

## Metrics and uncertainty

For raw and calibrated scores, calculate each metric per evaluation world,
then report the mean and a 95% percentile bootstrap interval over whole worlds
(500 resamples, seed 0). With only eight worlds these intervals are descriptive.

- RMSE against exact probabilities across the six base queries.
- Soft-target binary log loss across those queries; its optimum is target
  entropy, not necessarily zero. This is not an empirical calibration proof.
- Absolute complement, conjunction, and direct-versus-Bayes discrepancies for
  the base queries. A zero-probability conditioning event is explicitly undefined.
- Mean absolute changes across all six queries for paraphrased and irrelevant
  contexts; also mean two-formulation population variance for paraphrases.
- Ordinary-world complement and stability metrics separately, without accuracy
  or calibration claims.

Compare RMSE with a constant 0.5 baseline and a constant fitted to the mean of
calibration targets. No baseline, temperature, or threshold uses evaluation
labels for fitting. Report any missing/invalid result as incomplete; draw no
continuation conclusion from such a run. Report undefined Bayesian paths.

## Prospective continuation criteria

Require at least four calibration worlds, eight evaluation worlds, no failed
queries, and no undefined evaluation Bayesian paths. All the following must hold:

- Calibrated mean per-world RMSE <= 0.10 and strictly better than both baselines.
- Calibrated mean complement, conjunction, and Bayesian-path errors <= 0.10 each.
- Calibrated mean paraphrase and irrelevant-context absolute changes <= 0.05 each.

These are provisional engineering gates, not scientific claims of universal
usefulness. Failure is a useful result. Passing would justify replication with
more cases, models, and verbalizers before adding memory or semantic extraction.
Do not tune this protocol on the evaluation results; any revision needs a new
held-out suite. Save the protocol hash, query hash, runtime settings, raw records,
and calibration provenance with the report.
