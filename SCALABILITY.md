# GOLLuM Scalability Optimizations

This document details the 10 scalability optimizations applied to the GOLLuM codebase to reduce GPU memory usage (VRAM), improve throughput, and prevent out-of-memory (OOM) errors at scale.

---

## Priority 1 — OOM Preventers (Critical)

### 1A. Gradient Checkpointing in LLMFeaturizer

**File:** `src/gollum/featurization/deep.py`
**Problem:** During LoRA fine-tuning, the full forward pass through the LLM stores all intermediate activations for backpropagation. For a 7B-parameter model, this consumes ~32 GB in float32 — often exceeding available VRAM.

**Fix:** After wrapping the model with `get_peft_model()`, we call `self.llm.gradient_checkpointing_enable()`. This recomputes activations during the backward pass instead of storing them, trading ~30% extra compute for a 4–8x reduction in activation memory.

**Impact:** Enables fine-tuning of larger LLMs (e.g., Qwen-7B, Llama-8B) on GPUs with 16–24 GB VRAM that would otherwise OOM.

---

### 1B. Load LLMs in bfloat16 by Default

**Files:** `src/gollum/featurization/text.py`, `src/gollum/featurization/deep.py`

**Problem:** Models were loaded in float32 by default. `get_model_and_tokenizer()` used no dtype argument, and `LLMFeaturizer.__init__` explicitly cast to `torch.float32`. A 7B model in float32 = ~28 GB VRAM; in bfloat16 = ~14 GB.

**Fix:**
- `get_model_and_tokenizer()`: passes `torch_dtype=torch.bfloat16` to `AutoModel.from_pretrained()` for large models (known small models like T5 stay in float32).
- `LLMFeaturizer`: loads the LLM in bfloat16 on CUDA. The final output is still cast to float64 for GP numerical stability.

**Impact:** Halves VRAM for model parameters. Combined with gradient checkpointing (1A), enables 7B models on a single 24 GB GPU.

---

### 1C. Chunked Acquisition Function Evaluation

**File:** `src/gollum/bo/optimizer.py`

**Problem:** `optimize_acquisition_function()` evaluated the acquisition function over the **entire** design space in one call (`self.acquisition_function(design_space.unsqueeze(-2))`). For large candidate pools (10K+ points), this creates massive intermediate tensors on GPU.

**Fix:** Evaluate in chunks of 512 candidates at a time, collect acquisition values on CPU, then select the best index. Peak GPU memory is now bounded regardless of design space size.

**Impact:** Prevents OOM during the acquisition optimization step when the held-out pool is large.

---

### 1D. Batched Tokenization in `get_tokens()`

**File:** `src/gollum/featurization/text.py`

**Problem:** The entire dataset was tokenized in a single call to `tokenizer(texts, ...)`, creating the full padded tensor at once. Then `pad_sequence()` + `torch.cat()` created 2 additional full-size copies. For 10K sequences × 512 tokens, this tripled memory usage.

**Fix:** Tokenize in batches of `batch_size` texts. Each batch is padded to a fixed width (512) and collected. A single `torch.cat()` at the end produces the final tensor.

**Impact:** Reduces peak tokenization memory from 3× dataset size to ~1× dataset size + 1 batch.

---

## Priority 2 — Speed & Memory Leaks (High Impact)

### 2A. Model Loading Cache

**File:** `src/gollum/featurization/text.py`

**Problem:** `get_huggingface_embeddings()` called `get_model_and_tokenizer()` on every invocation, reloading the model from disk/cache into VRAM each time. For large models, this wastes time and risks double-loading.

**Fix:** Added a module-level `_MODEL_CACHE: dict` keyed by `(model_name, device)`. Subsequent calls return the cached model/tokenizer pair.

**Impact:** Eliminates redundant model loading. First call takes full time; subsequent calls are instant.

---

### 2B. dtype Cast Moved Outside Batch Loop

**File:** `src/gollum/featurization/deep.py`

**Problem:** Inside `get_embeddings()`, both `x = x.to(dtype=torch.float32)` and `self.llm = self.llm.to(dtype=torch.float32)` were executed on **every batch iteration**. The model cast re-checks/re-copies weights every batch even though nothing changes.

**Fix:** The input `x` is cast once before the loop. The model dtype is set in `__init__` and stored as `self._llm_dtype`.

**Impact:** Removes per-batch overhead. For 100 batches, this saves 100 redundant dtype-check operations on the full model.

---

### 2C. GPU Memory Leak Fix in DeepGP.forward()

**File:** `src/gollum/surrogate_models/gp.py`

**Problem:** `self.finetuned = finetuned` stored the embedding tensor as a persistent instance attribute on the GPU. This tensor was never freed, accumulating across BO iterations and training steps.

**Fix:** Removed the `self.finetuned` assignment. The local `finetuned` variable is used directly by `mean_module()` and `covar_module()` in the lines immediately below, then freed when the function returns.

**Impact:** Prevents GPU memory from growing linearly with the number of BO iterations.

---

### 2D. ProcessPoolExecutor Worker Limit

**File:** `src/gollum/featurization/text.py`

**Problem:** `ada_embeddings()` used `ProcessPoolExecutor()` with no `max_workers` argument. Python defaults to `os.cpu_count()` workers. On a 32-core machine, this spawns 32 processes, each holding a copy of the data.

**Fix:** Changed to `ProcessPoolExecutor(max_workers=min(8, len(texts)))`.

**Impact:** Bounds RAM usage and avoids API rate-limit errors for OpenAI embeddings.

---

## Priority 3 — Architecture-Level Improvements

### 3A. Pre-allocated Embedding Output Arrays

**Files:** `src/gollum/featurization/text.py`, `src/gollum/featurization/deep.py`

**Problem:** Both `get_huggingface_embeddings()` and `get_embeddings()` used `list.append()` followed by `np.concatenate()` / `torch.cat()`. Each concatenation copies all previous data into a new allocation, causing O(N/2) redundant copies for N batches.

**Fix:** Pre-allocate a single output array/tensor of shape `(N, embedding_dim)` after the first batch. Write each batch result in-place by slice index.

**Impact:** Eliminates quadratic memory overhead from repeated concatenation. For 10K samples, this saves ~5000 unnecessary array copies.

---

### 3B. Single-pass DataFrame Reindex

**File:** `src/gollum/data/module.py`

**Problem:** `split_data()` created `train_df.copy()`, `heldout_df.copy()`, then `pd.concat([train_df, heldout_df])` — 3 DataFrames in memory simultaneously (≈3× dataset size).

**Fix:** Replaced with `self.data = self.data.loc[ordered_indices].reset_index(drop=True)` — a single reindex operation with no intermediate copies.

**Impact:** Reduces peak memory during `setup()` from 3× to 1× dataset size.

---

## Additional Robustness Fixes

### Lazy OpenAI Client

**File:** `src/gollum/featurization/text.py`

The `OpenAI()` client was instantiated at module import time, crashing all imports when `OPENAI_API_KEY` was not set — even if only using HuggingFace models. Replaced with a lazy `_get_openai_client()` singleton that creates the client on first use.

### Lazy Reaction Featurizer Imports

**File:** `src/gollum/featurization/base.py`

The `rxnfp` package import in `_build_registry()` crashed if the package wasn't installed, even when not using reaction featurizers. Wrapped in `try/except ImportError` so reaction featurizers are optional.

### Guarded wandb.log() Calls

**File:** `src/gollum/surrogate_models/gp.py`

`DeepGP.forward()` called `wandb.log()` unconditionally. Added `if wandb.run is not None:` guard so the model works without an active wandb session (e.g., during benchmarking or testing).

---

## Summary Table

| # | Fix | File | Memory Impact | Speed Impact |
|---|-----|------|---------------|--------------|
| 1A | Gradient checkpointing | `deep.py` | 4–8x less activation memory | ~30% slower training |
| 1B | bfloat16 loading | `text.py`, `deep.py` | 2x less model VRAM | Neutral |
| 1C | Chunked acquisition | `optimizer.py` | Bounded peak VRAM | Neutral |
| 1D | Batched tokenization | `text.py` | 3x → 1x tokenization memory | Neutral |
| 2A | Model cache | `text.py` | No redundant loads | Fast on repeat calls |
| 2B | dtype cast outside loop | `deep.py` | Marginal | Faster per-batch |
| 2C | Memory leak fix | `gp.py` | Prevents linear growth | — |
| 2D | Worker limit | `text.py` | Bounded process RAM | — |
| 3A | Pre-allocated arrays | `text.py`, `deep.py` | No quadratic copies | Faster embedding |
| 3B | DataFrame reindex | `module.py` | 3x → 1x peak memory | Faster setup |

---

## Benchmarking

Run the benchmark suite to measure improvements:

```bash
# On main branch (baseline):
git checkout main
python benchmark/run_benchmarks.py

# On scale branch (optimized):
git checkout scale
python benchmark/run_benchmarks.py

# Compare:
python benchmark/analysis/summarize.py   # statistical tests
python benchmark/analysis/plots.py       # generates figures
```

Five experiments are included:

1. **Peak VRAM** — Wilcoxon signed-rank test on peak GPU memory
2. **Iteration Time** — Paired t-test on per-iteration wall-clock time
3. **Throughput Scaling** — Power-law fit on samples/sec vs. dataset size
4. **BO Quality** — Bonferroni-corrected paired t-test (null: no quality change)
5. **OOM Threshold** — Maximum dataset size before OOM
