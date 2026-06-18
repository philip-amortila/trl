#!/usr/bin/env bash
set -e

export WANDB_ENTITY=viano
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=2

cd ~/trl/trl/experimental/gkd

export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"

export TEACHER_MODEL=google/flan-t5-xl
export STUDENT_MODEL=google/flan-t5-large

export TRAIN_SAMPLES=7473
export EVAL_SAMPLES=200

export MAX_STEPS=500
export BATCH_SIZE=4
export GRAD_ACCUM=8
export LR=1e-4
export BETA=0.5
export LMBDA=1.0
export MAX_NEW_TOKENS=200

python -u gkd_t5_experiment.py 2>&1 | tee "run_gkd_S-flan-t5-large_T-flan-t5-xl_$(date +%Y%m%d_%H%M%S).log"
