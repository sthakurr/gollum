#!/bin/bash
#SBATCH --job-name=gollum-flip2-sweep
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=a131
#SBATCH --array=0-14         # 3 models x 5 seeds = 15 parallel runs

# --- Adjust these to your setup ---
CONDA_BASE="/iopsstor/scratch/cscs/ssaumya/piflow/miniforge3"
CONDA_ENV="gollum"
CONFIG="configs/flip2_pllmphi.yaml"
DATA_PATH="data/flip2/one_to_many_train.csv"   # training data (no validation)
FULL_DATA="data/flip2/one_to_many.csv"
SEEDS=(23 43 44 45 46)
MODELS=("Rostlab/prot_t5_xl_uniref50" "facebook/esm2_t33_650M_UR50D" "t5-base")
SWEEP_GROUP="flip2-gollum-sweep"
# -----------------------------------

N_SEEDS=${#SEEDS[@]}
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

set -euo pipefail

mkdir -p logs

echo "Array job:  $SLURM_ARRAY_JOB_ID  task: $SLURM_ARRAY_TASK_ID"
echo "Model:      $MODEL"
echo "Data:       $DATA_PATH"
echo "Seed:       $SEED"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:       $(date)"
echo "---"

# Activate conda
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "CUDA:   $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Patch model_name into a temp config (validate_configuration auto-adjusts input_dim)
TMP_CONFIG=$(mktemp /tmp/gollum_config_XXXXXX.yaml)
python3 - <<PYEOF
import yaml
with open("$CONFIG") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["init_args"]["featurizer"]["init_args"]["model_name"] = "$MODEL"
cfg["surrogate_model"]["init_args"]["finetuning_model"]["init_args"]["model_name"] = "$MODEL"
cfg["data"]["init_args"]["data_path"] = "$DATA_PATH"
with open("$TMP_CONFIG", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
PYEOF

# Run training — all tasks share the same group so wandb aggregates them
python train.py \
    --config "$TMP_CONFIG" \
    --visualize true \
    --full_data_path "$FULL_DATA" \
    --seed "$SEED" \
    --group "$SWEEP_GROUP"

rm -f "$TMP_CONFIG"

echo "--- Done (model=$MODEL, seed=$SEED): $(date) ---"
