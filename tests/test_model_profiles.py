import pytest
from probabilistic_oracle.backends import vllm


def test_default_engine_is_unchanged():
    assert vllm.engine_config() == dict(
        model=vllm.MODEL, revision=vllm.REVISION, tokenizer_revision=vllm.REVISION,
        dtype='bfloat16', max_model_len=512, max_num_seqs=1,
        max_num_batched_tokens=512, gpu_memory_utilization=0.5,
        enforce_eager=True, enable_prefix_caching=False, max_logprobs=-1,
        logprobs_mode='raw_logprobs', generation_config='vllm', seed=0,
    )


def test_larger_profile_pins_model_and_tokenizer_together():
    config = vllm.engine_config('qwen3-4b-awq')
    assert config['model'] == 'Qwen/Qwen3-4B-AWQ'
    assert config['revision'] == config['tokenizer_revision'] == '74d4bd2bd4bff9cafc9345221320bffb08b406a3'
    assert config['quantization'] == 'awq_marlin'
    assert config['dtype'] == 'float16'
    assert config['gpu_memory_utilization'] == 0.65
    config['model'] = 'changed'
    assert vllm.engine_config('qwen3-4b-awq')['model'] == 'Qwen/Qwen3-4B-AWQ'


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match='Unknown model profile'):
        vllm.engine_config('typo')
