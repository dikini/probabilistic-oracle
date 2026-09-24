"""Probability operations with explicit conditioning failures."""

import math


class UndefinedConditioning(ValueError):
    """The conditioning event has zero probability."""


def probability(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Expected a finite probability in [0, 1], got {value!r}")
    return float(value)


def complement(p: float) -> float:
    return 1 - probability(p)


def conjunction(p_a: float, p_b_given_a: float) -> float:
    return probability(p_a) * probability(p_b_given_a)


def bayes(prior: float, likelihood: float, alternative_likelihood: float) -> float:
    prior, likelihood, alternative_likelihood = map(probability, (prior, likelihood, alternative_likelihood))
    # Log-space products preserve very small nonzero evidence probabilities.
    def log_product(a, b):
        return math.log(a) + math.log(b) if a and b else -math.inf
    yes = log_product(prior, likelihood)
    no = log_product(1 - prior, alternative_likelihood)
    offset = max(yes, no)
    if offset == -math.inf:
        raise UndefinedConditioning("Evidence has zero probability under both branches")
    a, b = math.exp(yes - offset), math.exp(no - offset)
    return a / (a + b)
