import math
import os
from dataclasses import asdict
import json
from pathlib import Path
import pytest
from probabilistic_oracle.oracle import ScoreRequest

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get('RUN_GPU_TESTS') != '1', reason='Set RUN_GPU_TESTS=1 to use the GPU')]


def test_real_candidate_scores_and_provenance():
    from probabilistic_oracle.backends.vllm import VllmOracle
    oracle = VllmOracle()
    result = oracle.score(ScoreRequest('The box contains only red balls.', 'A ball drawn from the box is red.'))
    assert len(result.raw_logprobs) == 2
    assert all(math.isfinite(x) and x <= 0 for x in result.raw_logprobs)
    assert 0 <= result.score <= 1
    assert 0 < result.candidate_mass <= 1 + 1e-5
    assert result.provenance['backend'] == 'vllm'
    assert result.provenance['engine']['revision'] == 'c1899de289a04d12100db370d81485cdf75e47ca'
    Path('runs').mkdir(exist_ok=True)
    Path('runs/integration-score.json').write_text(json.dumps(asdict(result), indent=2))
