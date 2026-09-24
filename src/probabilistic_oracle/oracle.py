"""Scoring records and prompt validation, independent of the inference runtime."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ScoreRequest:
    context: str
    proposition: str
    assumptions: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()
    verbalizers: tuple[str, str] = ("TRUE", "FALSE")

    def __post_init__(self):
        if not self.context.strip() or not self.proposition.strip():
            raise ValueError("Context and proposition must be nonempty")
        if len(self.verbalizers) != 2 or len(set(self.verbalizers)) != 2 or any(not x.strip() for x in self.verbalizers):
            raise ValueError("Provide two distinct nonempty verbalizers, affirmative first")


@dataclass(frozen=True)
class ScoreResult:
    request: ScoreRequest
    prompt: str
    prompt_token_ids: list[int]
    candidate_token_ids: list[int]
    raw_logprobs: list[float]
    score: float
    candidate_mass: float
    generated_token_ids: list[int]
    seconds: float
    provenance: dict


def normalize(yes_logprob: float, no_logprob: float) -> tuple[float, float]:
    values = (yes_logprob, no_logprob)
    if any(not math.isfinite(x) or x > 0 for x in values):
        raise ValueError("Candidate log-probabilities must be finite and nonpositive")
    mass = sum(math.exp(x) for x in values)
    if mass > 1 + 1e-5:
        raise ValueError("Distinct candidate probabilities exceed unit mass")
    offset = max(values)
    a, b = (math.exp(x - offset) for x in values)
    return a / (a + b), mass


def extract_scores(logprobs: dict[int, float], candidate_ids: list[int]) -> tuple[float, float]:
    if len(candidate_ids) != 2 or len(set(candidate_ids)) != 2:
        raise ValueError("Expected two distinct candidate token IDs")
    if any(token not in logprobs for token in candidate_ids):
        raise ValueError("Missing candidate token log-probability")
    return normalize(*(logprobs[token] for token in candidate_ids))


def prepare_prompt(tokenizer, request: ScoreRequest) -> tuple[str, list[int], list[int]]:
    yes, no = request.verbalizers
    text = (
        f"Assess the proposition given the context. Answer with exactly {yes} or {no}.\n"
        f"Use {yes} for true and {no} for false.\nContext: {request.context}\n"
    )
    if request.assumptions:
        text += "Assumptions (suppose these hold):\n" + "\n".join(request.assumptions) + "\n"
    if request.observations:
        text += "Observations (new evidence):\n" + "\n".join(request.observations) + "\n"
    text += f"Proposition: {request.proposition}"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    candidates = []
    for label in request.verbalizers:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Verbalizer {label!r} is not a single token")
        if tokenizer.encode(prompt + label, add_special_tokens=False) != prompt_ids + ids:
            raise ValueError(f"Verbalizer {label!r} changes tokenization at the prompt boundary")
        candidates.append(ids[0])
    if len(set(candidates)) != 2:
        raise ValueError("Verbalizers map to the same token")
    return prompt, prompt_ids, candidates
