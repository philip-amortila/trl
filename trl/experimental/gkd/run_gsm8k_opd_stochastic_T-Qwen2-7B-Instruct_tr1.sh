#!/usr/bin/env bash
set -e

export WANDB_ENTITY=viano
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=3

cd ~/trl/trl/experimental/gkd

# ------------------------
# Auth
# ------------------------
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
python -c "import wandb; wandb.login()"

export TEACHER_MODEL=Qwen/Qwen2-7B-Instruct
export STUDENT_MODEL=Qwen/Qwen2-0.5B-Instruct

export TRAIN_SAMPLES=2000
export EVAL_SAMPLES=256

export MAX_STEPS=800
export BATCH_SIZE=1
export GRAD_ACCUM=8
export LR=2e-6

export OPD_MODE=stochastic
export NUM_INNER_STEPS=10
export REPLAY_BUFFER_SIZE=10
export TRUST_REGION=1

python -u opd_experiment.py 2>&1 | tee "run_opd_stochastic_T-Qwen2-7B-Instruct_tr1_$(date +%Y%m%d_%H%M%S).log"
