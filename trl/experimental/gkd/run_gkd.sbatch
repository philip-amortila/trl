#!/usr/bin/env bash
#SBATCH -p songmei
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -J gkd
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

source ~/.bashrc
conda activate opd

# Put HF caches on scratch (avoids home quota issues)
export HF_HOME=/scratch/users/p.amortila/hf
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

# W&B: store local run files on scratch
export WANDB_DIR=/scratch/users/p.amortila/wandb
mkdir -p "$WANDB_DIR"

# Optional but often helpful on clusters
export TOKENIZERS_PARALLELISM=false

cd ~/trl/trl/experimental/gkd

# If you want the run to show up under a specific W&B project:
# export WANDB_PROJECT=trl_gkd
# export WANDB_ENTITY=<your_wandb_username_or_team>

python -u my_opd.py
