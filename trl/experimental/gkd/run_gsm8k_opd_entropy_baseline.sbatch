#!/usr/bin/env bash
#SBATCH -p songmei
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 6:00:00
#SBATCH -J gsm8k_opd_entropy_baseline
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

source ~/.bashrc
conda activate opd

# ------------------------
# HuggingFace cache → scratch
# ------------------------
export HF_HOME=/scratch/users/p.amortila/hf
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

# ------------------------
# Disable W&B for now
# ------------------------
export WANDB_MODE=disabled

# Prevent tokenizer fork warnings
export TOKENIZERS_PARALLELISM=false

cd ~/trl/trl/experimental/gkd

# ------------------------
# Experimental knobs
# (edit here without touching python file)
# ------------------------
export STUDENT_MODEL=Qwen/Qwen2-0.5B-Instruct
export TEACHER_MODEL=Qwen/Qwen2-1.5B-Instruct

export TRAIN_SAMPLES=2000
export EVAL_SAMPLES=256

export MAX_STEPS=800
export BATCH_SIZE=1
export GRAD_ACCUM=8
export LR=2e-6

# OPD loss mode: "entropy_baseline" | "expectation" | "stochastic"
# For entropy_baseline: advantage = -H(pi_e) - log pi_e(a_s), REINFORCE gradient
# For expectation:      advantage = log pi_e(a_e) - E_{a~pi_s}[log pi_e(a)], analytical gradient
export OPD_MODE=entropy_baseline
export OUTPUT_DIR=opd_gsm8k_out_${OPD_MODE}_$(date +%Y%m%d_%H%M%S)

# ------------------------
# Run experiment
# ------------------------
python -u opd_experiment.py
