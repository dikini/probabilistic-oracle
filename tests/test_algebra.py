import math
import pytest
from probabilistic_oracle.algebra import complement, conjunction, bayes, UndefinedConditioning


def test_reference_probability_operations():
    assert complement(0.2) == pytest.approx(0.8)
    assert conjunction(0.2, 0.8) == pytest.approx(0.16)
    assert bayes(0.2, 0.8, 0.1) == pytest.approx(2 / 3)


@pytest.mark.parametrize('prior,expected', [(0, 0), (1, 1)])
def test_extreme_prior(prior, expected):
    assert bayes(prior, 0.8, 0.1) == expected


def test_impossible_evidence():
    with pytest.raises(UndefinedConditioning):
        bayes(0.2, 0, 0)


def test_tiny_likelihoods_do_not_underflow_to_undefined():
    assert bayes(0.2, 1e-320, 1e-320) == pytest.approx(0.2)


@pytest.mark.parametrize('bad', [-0.1, 1.1, math.nan, math.inf, -math.inf, True])
def test_invalid_probability_inputs(bad):
    with pytest.raises(ValueError):
        complement(bad)
    for args in [(bad, .2), (.2, bad)]:
        with pytest.raises(ValueError):
            conjunction(*args)
    for args in [(bad, .2, .3), (.2, bad, .3), (.2, .3, bad)]:
        with pytest.raises(ValueError):
            bayes(*args)
