#!/bin/bash
# ============================================================
# FLIP2 Benchmark: all models x all 4 enzyme datasets x seeds
#
# Models   : ESMC-600M, ESM2-650M, ProtT5-XL, t5-base
# Datasets : alpha-amylase, ired, nucB, trpB
# Seeds    : 5 per combination
# Total    : 4 models x 5 datasets x 5 seeds = 100 array tasks
#
# ESM3 note: add "EvolutionaryScale/esm3-sm-open-v1" to MODELS
# and ensure the esm package supports it (needs _is_esm3_model
# in featurization/text.py analogous to _is_esmc_model).
#
# Submit: sbatch jobs/flip2_benchmark.sh
# ============================================================
#SBATCH --job-name=flip2-bench
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=a131
#SBATCH --array=0-74        # 3 models x 5 datasets x 5 seeds

# ---- Adjust these to your setup ----
CONDA_BASE="/users/ssaumya/miniforge3"
CONDA_ENV="gollum"
SWEEP_GROUP="gollum_flip2_benchmark_v1"
WANDB_PROJECT="gollum_flip2"
# ------------------------------------

# Models: ESMC uses flip2_pllmphi.yaml (esm tokenizer); HF models use flip2_hf.yaml
MODELS=(
    "facebook/esm2_t33_650M_UR50D"           # 1  ESM2  – hf config
    "Rostlab/prot_t5_xl_uniref50"            # 2  ProtT5 – hf config
    "t5-base"                                # 3  t5-base – hf config
)

# Which base config each model uses
MODEL_CONFIGS=(
    "configs/flip2_hf.yaml"        # ESM2
    "configs/flip2_hf.yaml"        # ProtT5
    "configs/flip2_hf.yaml"        # t5-base
)

# Pooling method for the static featurizer (used only for HF models in get_tokens mode)
MODEL_POOLING=(
    "average"   # ESM2
    "average"   # ProtT5
    "average"   # t5-base
)

# Dataset names and their training-split CSV names
DATASETS=("alpha-amylase/one_to_many_train.csv"    "ired/two_to_many_train.csv"            "nucB/two_to_many_train.csv"            "trpB/one_to_many_train.csv"     "alpha-amylase/close_to_far_train.csv")
#SPLITS=(  "one_to_many"      "two_to_many"     "two_to_many"     "one_to_many")

SEEDS=(1 15 23 43 50)

N_MODELS=${#MODELS[@]}
N_DATASETS=${#DATASETS[@]}
N_SEEDS=${#SEEDS[@]}

# Decompose flat task ID → (model, dataset, seed)
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / (N_DATASETS * N_SEEDS) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (N_DATASETS * N_SEEDS) ))
DATASET_IDX=$(( REMAINDER / N_SEEDS ))
SEED_IDX=$(( REMAINDER % N_SEEDS ))

MODEL="${MODELS[$MODEL_IDX]}"
BASE_CONFIG="${MODEL_CONFIGS[$MODEL_IDX]}"
POOLING="${MODEL_POOLING[$MODEL_IDX]}"
DATASET="${DATASETS[$DATASET_IDX]}"
# SPLIT="${SPLITS[$DATASET_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
DATA_PATH="data/flip2/${DATASET}"

set -euo pipefail
mkdir -p logs

echo "========================================"
echo "Array task : $SLURM_ARRAY_JOB_ID / $SLURM_ARRAY_TASK_ID"
echo "Model      : $MODEL"
echo "Dataset    : $DATASET  ($DATA_PATH)"
echo "Seed       : $SEED"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date       : $(date)"
echo "========================================"

# Activate conda
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="/capstor/store/cscs/swissai/a131/ssaumya/.cache/huggingface"

# Build a temporary config with model_name, input_dim, pooling, and data_path patched
TMP_CONFIG=$(mktemp /tmp/gollum_flip2_XXXXXX.yaml)

python3 - <<PYEOF
import yaml, sys

MODEL      = "$MODEL"
DATA_PATH  = "$DATA_PATH"
POOLING    = "$POOLING"

with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)

# ---- Patch model names ----
cfg["data"]["init_args"]["featurizer"]["init_args"]["model_name"] = MODEL
cfg["data"]["init_args"]["featurizer"]["init_args"]["pooling_method"] = POOLING
cfg["data"]["init_args"]["data_path"] = DATA_PATH

ft = cfg["surrogate_model"]["init_args"]["finetuning_model"]["init_args"]
ft["model_name"] = MODEL
ft["pooling_method"] = POOLING

# ---- Auto-set input_dim from known embedding sizes ----
EMBEDDING_SIZES = {
    "EvolutionaryScale/esmc-600m-2024-12": 1152,
    "facebook/esm2_t33_650M_UR50D":        1280,
    "Rostlab/prot_t5_xl_uniref50":         1024,
    "t5-base":                             768,
    "EvolutionaryScale/esm3-sm-open-v1":   1024,
}
if MODEL in EMBEDDING_SIZES:
    ft["input_dim"] = EMBEDDING_SIZES[MODEL]

cfg["wandb_project"] = "$WANDB_PROJECT"

with open("$TMP_CONFIG", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print(f"Config written to $TMP_CONFIG  (model={MODEL}, dataset=$DATASET, seed=$SEED)")
PYEOF

# Run training
python train.py \
    --config  "$TMP_CONFIG" \
    --seed    "$SEED" \
    --group   "flip2_v1_${DATASET%%/*}"

rm -f "$TMP_CONFIG"
echo "--- Done (model=$MODEL, dataset=$DATASET, seed=$SEED): $(date) ---"
