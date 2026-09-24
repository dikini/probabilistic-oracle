# Working on Probabilistic Oracle

Read README.md and the relevant milestone documents before changing code.
Distinguish agreed scope, proposed implementation details, and measured results.

Keep the first milestone limited to oracle, algebra, and bench. Treat token
scores as plausibility scores until calibration evidence supports a stronger
interpretation. Keep raw model observations separate from calibrated values and
algebraically derived values.

Use frozen Hugging Face models and vLLM. Verify the selected versions' scoring
behavior before relying on it. Record model/tokenizer revisions, prompts,
verbalizers, inference settings, and calibration provenance with experiments.
Never substitute missing token probabilities with zero or silently use a mock
backend for reported model results.

Keep tests for algebra and benchmark logic runnable without a GPU. Validate real
backend behavior separately. Document which checks ran and which require hardware.
Keep model weights, credentials, caches, and generated run outputs out of Git.
Do not choose a license, publish a remote, or download large models without an
explicit request. Write plainly and report negative findings as results.
