#!/usr/bin/env bash
#SBATCH -p songmei
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J eval_gsm8k
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

source ~/.bashrc
conda activate opd

export HF_HOME=/scratch/users/p.amortila/hf
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

cd ~/trl/trl/experimental/gkd

# ------------------------
# Knobs (override via env or edit here)
# ------------------------
export MODEL_DIR=${MODEL_DIR:-"gkd_gsm8k_out"}
export EVAL_N=${EVAL_N:-"1319"}

# ------------------------
# Run eval (always reports both strict and flexible)
# ------------------------
python -u eval_gsm8k.py \
    --model_dir "$MODEL_DIR" \
    --n "$EVAL_N"
