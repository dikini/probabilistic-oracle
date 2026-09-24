"""Bounded hardware/scoring probe; not a calibration benchmark.

Run from the repository root with .venv/bin/python scripts/gpu_smoke.py.
"""

import json
import math
import os
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ENGINE = dict(
    model=MODEL,
    revision=REVISION,
    tokenizer_revision=REVISION,
    dtype="bfloat16",
    max_model_len=512,
    max_num_seqs=1,
    max_num_batched_tokens=512,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
    enable_prefix_caching=False,
    max_logprobs=-1,
    logprobs_mode="raw_logprobs",
    generation_config="vllm",
    seed=0,
)


def gpu_snapshot():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=5
    ).strip()


def main():
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    output = Path("runs") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-gpu-smoke.json")
    output.parent.mkdir(exist_ok=True)
    record = {
        "purpose": "GPU fit and raw token-logprob extraction, not calibration",
        "model": MODEL, "revision": REVISION, "engine": ENGINE,
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ("vllm", "torch", "transformers", "huggingface-hub")},
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": subprocess.check_output(["git", "status", "--short"], text=True),
        "gpu_before": gpu_snapshot(), "gpu_samples": [], "cases": [],
    }
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                record["gpu_samples"].append(gpu_snapshot())
            except Exception as exc:
                record.setdefault("monitor_errors", []).append(str(exc))
            stop.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        labels = {label: tokenizer.encode(label, add_special_tokens=False) for label in ("TRUE", "FALSE")}
        assert all(len(ids) == 1 for ids in labels.values()), labels
        record["verbalizers"] = labels
        start = time.perf_counter()
        llm = LLM(**ENGINE)
        record["engine_start_seconds"] = time.perf_counter() - start
        # Full-vocabulary logprobs ensure both candidates exist in this version.
        params = SamplingParams(temperature=0, max_tokens=1, logprobs=-1)
        record["sampling"] = {"temperature": 0, "max_tokens": 1, "logprobs": -1}
        cases = [
            ("certain_true", "The box contains only red balls. One ball is drawn.", "The drawn ball is red."),
            ("certain_false", "The box contains only red balls. One ball is drawn.", "The drawn ball is not red."),
            ("fair_coin", "A fair coin is tossed once.", "The coin lands heads."),
            ("fair_coin_negated", "A fair coin is tossed once.", "The coin does not land heads."),
        ]
        for case_id, context, proposition in cases:
            messages = [{"role": "user", "content": (
                "Assess the proposition given the context. Answer with exactly TRUE or FALSE.\n"
                f"Context: {context}\nProposition: {proposition}"
            )}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            for label, ids in labels.items():
                assert tokenizer.encode(prompt + label, add_special_tokens=False) == prompt_ids + ids
            start = time.perf_counter()
            result = llm.generate([{"prompt_token_ids": prompt_ids}], params, use_tqdm=False)[0]
            elapsed = time.perf_counter() - start
            generated = result.outputs[0]
            values = {label: generated.logprobs[0][ids[0]].logprob for label, ids in labels.items()}
            assert all(math.isfinite(x) and x <= 0 for x in values.values()), values
            offset = max(values.values())
            scaled = {k: math.exp(v - offset) for k, v in values.items()}
            score = scaled["TRUE"] / sum(scaled.values())
            case = dict(id=case_id, context=context, proposition=proposition,
                        prompt=prompt, prompt_token_ids=prompt_ids,
                        generated_text=generated.text, generated_token_ids=generated.token_ids,
                        raw_logprobs=values, true_score=score,
                        candidate_mass=sum(math.exp(v) for v in values.values()),
                        seconds=elapsed)
            record["cases"].append(case)
            print(json.dumps({k: case[k] for k in ("id", "true_score", "candidate_mass", "seconds")}), flush=True)
        record["gpu_loaded"] = gpu_snapshot()
        record["status"] = "passed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        stop.set()
        thread.join(timeout=6)
        output.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Run record: {output}", flush=True)


if __name__ == "__main__":
    main()
