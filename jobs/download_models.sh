#!/bin/bash
# ============================================================
# Pre-download all models needed for flip2_benchmark.sh
# Run this ONCE on a login node (internet access) before
# submitting the benchmark array job.
#
# Usage: bash jobs/download_models.sh
# ============================================================

CONDA_BASE="/users/ssaumya/miniforge3"
CONDA_ENV="gollum"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HOME="/capstor/store/cscs/swissai/a131/ssaumya/.cache/huggingface"

# Unset offline flags so downloads can happen
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

python3 - <<'PYEOF'
from transformers import AutoTokenizer, AutoModel, T5Tokenizer
import torch

# MODELS = {
#     # model_id: (loader, kwargs)
#     "facebook/esm2_t33_650M_UR50D": ("hf", {}),
#     "Rostlab/prot_t5_xl_uniref50":  ("prot_t5", {}),
#     "t5-base":                       ("hf", {}),
# }

# for model_id, (loader, kwargs) in MODELS.items():
#     print(f"\n=== Downloading {model_id} ===")
#     try:
#         if loader == "prot_t5":
#             T5Tokenizer.from_pretrained(model_id, do_lower_case=False, legacy=True)
#             AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
#         else:
#             AutoTokenizer.from_pretrained(model_id)
#             AutoModel.from_pretrained(model_id)
#         print(f"    OK: {model_id}")
#     except Exception as e:
#         print(f"    FAILED: {model_id}: {e}")

# ESMC requires the esm package (EvolutionaryScale)
print("\n=== Downloading ESMC-600M via esm package ===")
try:
    from esm.models.esmc import ESMC
    ESMC.from_pretrained("esmc_600m")
    print("    OK: esmc_600m")
except Exception as e:
    print(f"    FAILED (esmc_600m): {e}")

# ESM3 (optional – uncomment when featurizer support is added)
# print("\n=== Downloading ESM3-open via esm package ===")
# try:
#     from esm.models.esm3 import ESM3
#     ESM3.from_pretrained("esm3-sm-open-v1")
#     print("    OK: esm3-sm-open-v1")
# except Exception as e:
#     print(f"    FAILED: {e}")

print("\nAll downloads complete. You can now submit with TRANSFORMERS_OFFLINE=1.")
PYEOF
