import math
import pytest
from probabilistic_oracle.oracle import ScoreRequest, normalize, prepare_prompt


def test_normalization_preserves_low_candidate_mass():
    score, mass = normalize(-1000, -1001)
    assert score == pytest.approx(0.73105857863)
    assert mass == 0  # Raw logs, retained separately, remain the source of truth.
    score, mass = normalize(math.log(.3), math.log(.1))
    assert score == pytest.approx(.75)
    assert mass == pytest.approx(.4)


@pytest.mark.parametrize('values', [(math.nan, -1), (-1, math.inf), (-math.inf, -1), (.1, -1), (0, 0)])
def test_reject_invalid_or_impossible_candidate_logprobs(values):
    with pytest.raises(ValueError):
        normalize(*values)


class BoundaryTokenizer:
    """Minimal tokenizer boundary double; no inference is simulated."""
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content'] + '\nASSISTANT:'

    def encode(self, text, add_special_tokens=False):
        if text in ('TRUE', 'FALSE'):
            return [1 if text == 'TRUE' else 2]
        for suffix, token in [('TRUE', 1), ('FALSE', 2)]:
            if text.endswith('ASSISTANT:' + suffix):
                return [10, 11, token]
        return [10, 11]


def test_prompt_records_assumptions_and_observations_separately():
    req = ScoreRequest('base context', 'H', assumptions=('A',), observations=('E',))
    prompt, ids, candidates = prepare_prompt(BoundaryTokenizer(), req)
    assert 'Assumptions (suppose these hold):\nA' in prompt
    assert 'Observations (new evidence):\nE' in prompt
    assert ids == [10, 11]
    assert candidates == [1, 2]


def test_verbalizer_boundary_merge_rejected():
    class MergingTokenizer(BoundaryTokenizer):
        def encode(self, text, **kwargs):
            if text.endswith('ASSISTANT:TRUE'):
                return [10, 99]
            return super().encode(text, **kwargs)
    with pytest.raises(ValueError, match='boundary'):
        prepare_prompt(MergingTokenizer(), ScoreRequest('context', 'H'))


@pytest.mark.parametrize('labels', [('TRUE','TRUE'), ('','FALSE'), ('TRUE',)])
def test_bad_verbalizer_pairs(labels):
    with pytest.raises(ValueError):
        ScoreRequest('context', 'H', verbalizers=labels)


def test_missing_candidate_does_not_become_zero():
    from probabilistic_oracle.oracle import extract_scores
    with pytest.raises(ValueError, match='Missing'):
        extract_scores({1: -.2}, [1, 2])
    assert extract_scores({1: math.log(.3), 2: math.log(.1)}, [1, 2]) == pytest.approx((.75, .4))


def test_multi_token_verbalizer_rejected():
    class MultiToken(BoundaryTokenizer):
        def encode(self,text,**kwargs):
            if text=='TRUE':
                return [1,4]
            return super().encode(text,**kwargs)
    with pytest.raises(ValueError,match='single token'):
        prepare_prompt(MultiToken(),ScoreRequest('context','H'))
