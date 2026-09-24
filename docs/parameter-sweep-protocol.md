# Bounded parameter sweep: 2026-09-24

Status: fixed before running the sweep. Overall session budget approximately
20 minutes; GPU work has an explicit UTC deadline supplied to the runner.
Reserve time for a single fresh evaluation and reporting. Keep the existing
continuation gates unchanged.

The previous eight evaluation worlds are now development data. Screen on the
first four of them, with all three variants and six queries (72 requests), plus
24 base requests from the four original calibration worlds. Reuse the previous
baseline observations transparently. Each new configuration needs 96 requests.

Try, in order: baseline prompt with YES/NO; baseline with A/B; random-outcome
instructions with TRUE/FALSE; frequency instructions with TRUE/FALSE; frequency
instructions with A/B; frequency examples with TRUE/FALSE. Stop adding candidates
when the remaining time cannot cover a candidate plus fresh evaluation. The
model revision, precision, context length, and GPU memory budget remain fixed.

Check decoding temperatures 0, 0.5, and 1 on one identical request. Since the
backend returns raw log-probabilities before sampling, these should not change
the measured distribution; do not waste a full grid on equivalent measurements.

For each candidate, fit global binary log-odds temperature and intercept on the
24 calibration base queries only, minimizing soft-target log loss. Temperature
grid: [0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 8, 16, 32]. Intercept grid:
[-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2]. Also report raw and temperature-only
performance. Choose the candidate with the smallest maximum normalized gate
violation on development worlds; break ties by calibrated RMSE. Fit no per-query
or per-case parameter and apply no probability-coherence projection.

Freeze the chosen configuration and calibration parameters to selection.json
before reading or scoring the new eight-world holdout. A deterministic generator
(seed 240924, positive counts summing to 100, excluding every existing world)
defines that holdout. Record its SHA-256 before inference. Evaluate it once, with
all variants and six queries (144 requests), and the four ordinary-world cases
(24 requests). Never revise a configuration using fresh evaluation results.

The final gates are the existing ones: RMSE <= 0.10 and better than both constant
baselines, complement/conjunction/Bayes errors <= 0.10, paraphrase and irrelevant
context changes <= 0.05, and no failed requests or undefined Bayesian paths.
Baseline fitted constant uses calibration targets only. Development successes
are tuning results; only the fresh holdout supplies a continuation decision.
Failure is an acceptable outcome, and no pass is promised within this budget.
