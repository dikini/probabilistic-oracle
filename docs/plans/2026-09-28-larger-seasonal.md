# Larger quantized model on the same seasonal tests

Compare the frozen Qwen3-4B-AWQ checkpoint (revision
74d4bd2bd4bff9cafc9345221320bffb08b406a3) against the completed Qwen3-0.6B
runs. Repeat the same 36 no-facts and 36 supplied-facts requests, including
wording, choices, order, greedy decoding, disabled thinking, token limit and
scoring. The facts run uses its own larger-model no-facts baseline.

The official checkpoint contains 2,666,027,672 bytes of safetensors weights,
4-bit AWQ with group size 128. Use vLLM 0.11.2 AWQ Marlin, FP16 activations,
512-token context, one sequence, eager execution and 65% GPU memory budget.
Keep the original 0.6B configuration unchanged. This compares deployable model
configurations; size, quantization and activation dtype are confounded. It does
not estimate quantization loss or demonstrate calibrated Bayesian priors.

No fitting or prompt selection. Report supplied-profile agreement, original
strict-label agreement, unknown answers, stability and seasonal contrasts, all
with their original denominators. Supplied labels remain task stipulations.
Report all case answers and compare exact request text with earlier manifests.
These are reused diagnostic cases, not a fresh holdout or broad benchmark.

Implementation: add pinned named backend profiles and a seasonal CLI selector;
verify defaults and profile selection with CPU tests. Commit unsigned before
inference. Download the pinned public checkpoint without remote custom code.
Allow at most 10 minutes for each inference process, monitor shared GPU memory,
and preserve failures. Only resource-related adaptations are permitted if needed,
with attempts recorded. Recompute saved reports, review, document and merge.
