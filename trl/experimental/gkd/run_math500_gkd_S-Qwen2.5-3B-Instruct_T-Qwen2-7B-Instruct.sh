#!/usr/bin/env bash
set -e

export WANDB_ENTITY=viano
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=5

cd ~/trl/trl/experimental/gkd

# ------------------------
# Auth
# ------------------------
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
python -c "import wandb; wandb.login()"

export TEACHER_MODEL=Qwen/Qwen2-7B-Instruct
export STUDENT_MODEL=Qwen/Qwen2.5-3B-Instruct

export TRAIN_SAMPLES=7500
export EVAL_SAMPLES=500

export MAX_STEPS=800
export BATCH_SIZE=1
export GRAD_ACCUM=8
export LR=2e-6
export BETA=0.5

python -u math500_gkd_experiment.py 2>&1 | tee "run_math500_gkd_S-Qwen2.5-3B-Instruct_T-Qwen2-7B-Instruct_$(date +%Y%m%d_%H%M%S).log"
