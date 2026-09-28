"""Adapter for the vLLM 0.11.2 raw-logprob contract verified on the laptop."""

from importlib.metadata import version
import os
import time

from probabilistic_oracle.oracle import ScoreRequest, ScoreResult, extract_scores, prepare_prompt

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


PROFILES = ('qwen3-0.6b', 'qwen3-4b-awq')


def engine_config(profile='qwen3-0.6b'):
    """Return a fresh pinned configuration; importing this does not load vLLM."""
    if profile not in PROFILES:
        raise ValueError(f'Unknown model profile: {profile}')
    config = dict(
        model=MODEL, revision=REVISION, tokenizer_revision=REVISION,
        dtype="bfloat16", max_model_len=512, max_num_seqs=1,
        max_num_batched_tokens=512, gpu_memory_utilization=0.5,
        enforce_eager=True, enable_prefix_caching=False,
        max_logprobs=-1, logprobs_mode="raw_logprobs",
        generation_config="vllm", seed=0,
    )
    if profile == 'qwen3-4b-awq':
        revision = '74d4bd2bd4bff9cafc9345221320bffb08b406a3'
        config.update(model='Qwen/Qwen3-4B-AWQ', revision=revision,
                      tokenizer_revision=revision, dtype='float16',
                      quantization='awq_marlin', gpu_memory_utilization=0.65)
    return config


class VllmOracle:
    def __init__(self, profile="qwen3-0.6b"):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        if version("vllm") != "0.11.2":
            raise RuntimeError("This adapter requires the verified vLLM 0.11.2 contract")
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self.engine = engine_config(profile)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.engine['model'], revision=self.engine['tokenizer_revision'])
        self.llm = LLM(**self.engine)
        self.sampling = dict(temperature=0, max_tokens=1, logprobs=-1)
        self.params = SamplingParams(**self.sampling)
        self.provenance = {
            "backend": "vllm", "engine": self.engine, "sampling": self.sampling,
            "prompt_template": "binary-v1", "enable_thinking": False,
            "versions": {p: version(p) for p in ("vllm", "torch", "transformers", "huggingface-hub")},
        }

    def score(self, request: ScoreRequest) -> ScoreResult:
        prompt, prompt_ids, candidates = prepare_prompt(self.tokenizer, request)
        if len(prompt_ids) + self.sampling["max_tokens"] > self.engine["max_model_len"]:
            raise ValueError("Prompt exceeds the configured context limit; it was not truncated")
        start = time.perf_counter()
        output = self.llm.generate([{"prompt_token_ids": prompt_ids}], self.params, use_tqdm=False)[0].outputs[0]
        if not output.logprobs:
            raise ValueError("Backend returned no token log-probabilities")
        # Only retain the two requested observations; no missing-token substitution.
        logs = {token: output.logprobs[0][token].logprob for token in candidates if token in output.logprobs[0]}
        score, mass = extract_scores(logs, candidates)
        return ScoreResult(
            request, prompt, prompt_ids, candidates, [logs[t] for t in candidates],
            score, mass, list(output.token_ids), time.perf_counter() - start,
            self.provenance,
        )
