#!/usr/bin/env bash
set -e

export WANDB_ENTITY=viano
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1

cd ~/trl/trl/experimental/gkd

export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"

export TEACHER_MODEL=google/flan-t5-xl
export STUDENT_MODEL=google/flan-t5-base

export TRAIN_SAMPLES=7473
export EVAL_SAMPLES=200

export MAX_STEPS=500
export BATCH_SIZE=4
export GRAD_ACCUM=8
export LR=1e-5
export LMBDA=1.0
export MAX_NEW_TOKENS=200

export OPD_MODE=expectation
export NUM_INNER_STEPS=10
export REPLAY_BUFFER_SIZE=10
export TRUST_REGION=1
export PPO_CLIP_EPS=0.2

python -u opd_t5_experiment.py 2>&1 | tee "run_opd_S-flan-t5-base_T-flan-t5-xl_$(date +%Y%m%d_%H%M%S).log"
