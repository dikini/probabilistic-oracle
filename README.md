# Probabilistic Oracle

A local experiment testing whether a frozen open-source transformer exposes a
stable, probability-like plausibility signal through its token logits.

The model supplies uncertain judgments. External code computes probability
identities and Bayesian updates. A normalized token score is a plausibility
score; interpreting it as a probability requires empirical evidence.

## First milestone

- **oracle:** one vLLM backend using a Hugging Face model, proposition scoring,
  and configurable binary verbalizers.
- **algebra:** negation, conjunction through conditional decomposition, Bayesian
  updates, and discrepancy metrics.
- **bench:** reproducible synthetic cases with known probabilities and a small
  collection of ordinary-world propositions.

Measure calibration, paraphrase variance, negation consistency, agreement between
direct and algebraic Bayesian updates, and sensitivity to irrelevant context.

Defer memory/RAG, semantic extraction, hypothesis generation, belief graphs,
coherence projection, decision procedures, hidden-state probes, and fine-tuning.

## Use

Run commands from this repository's root. Python 3.12 or newer is required.

For algebra, reports, and tests without a GPU:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest
.venv/bin/probabilistic-oracle validate
```

For the pinned Linux NVIDIA environment tested on the laptop:

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements-smoke.lock
uv pip install --python .venv/bin/python -e '.[test]'
RUN_GPU_TESTS=1 .venv/bin/python -m pytest tests/integration -q
.venv/bin/probabilistic-oracle run --output runs/first-benchmark
.venv/bin/probabilistic-oracle report runs/first-benchmark
```

A run directory must be new. Interrupted runs are retained and rejected as
incomplete by reporting; rerun in a new directory. Failed requests are recorded
explicitly, and an incomplete report exits with status 2. Failing the research
thresholds is a valid experimental result and exits with status 0. Reports include
raw and calibrated metrics, baselines, uncertainty across cases, and fit provenance.

`--verbalizers yes-no` or `--verbalizers a-b` selects an alternative pair for a
separate run. The first benchmark protocol uses TRUE/FALSE only.

The Python scoring API keeps inference observations separate from algebra:

```python
from probabilistic_oracle.algebra import bayes
from probabilistic_oracle.oracle import ScoreRequest
from probabilistic_oracle.backends.vllm import VllmOracle

# Put GPU initialization under this guard: vLLM spawns a worker process.
if __name__ == "__main__":
    oracle = VllmOracle()
    result = oracle.score(ScoreRequest(
        context="A fair coin is tossed once.",
        proposition="The coin lands heads.",
    ))
    print(result.score, result.raw_logprobs, result.candidate_mass)
    print(bayes(0.2, 0.8, 0.1))  # 2/3; exact external probability operation
```

## Evidence and scope

The initial GPU smoke test passed on an RTX 4050 Laptop GPU shared with the GUI.
Qwen3-0.6B in BF16 ran through vLLM with a 512-token context and a 50% GPU memory
budget. Total observed GPU usage peaked at 4,093 MiB, with 1,670 MiB reported free.
The package implements the scoring API, algebra, finite-world cases, calibration,
and reports. The first 240-request benchmark completed without failures but did
not meet its continuation criteria: calibrated probability RMSE was 0.2499,
slightly worse than both constant baselines, and conjunction error was 0.2410.
The measured setup is not yet a useful calibrated probability oracle.

- [Three- and five-category truth-scale results](docs/categorical-readout-2026-09-24.md)
- [Numerical versus binary readout comparison](docs/readout-comparison-2026-09-24.md)
- [Bounded parameter sweep](docs/parameter-sweep-2026-09-24.md)
- [First benchmark results](docs/benchmark-2026-09-24.md)
- [GPU smoke test](docs/gpu-smoke-2026-09-24.md)
- [Prospective experiment protocol](docs/experiment-protocol.md)
- [Milestone design](docs/plans/2026-09-24-milestone-1-design.md)
- [Implementation plan](docs/plans/2026-09-24-milestone-1.md)

Development is local and tracked in Git. Model weights, caches, credentials, and
run outputs stay outside version control. Runs retain exact prompts and raw
observations in `manifest.json` and `records.jsonl`; `report.json` is derived.
Do not interpret binary scores as calibrated probabilities without supporting
measurements, or treat consistency alone as evidence of accuracy.
