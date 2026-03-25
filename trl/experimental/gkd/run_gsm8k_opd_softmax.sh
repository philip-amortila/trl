#!/usr/bin/env bash
set -e

export WANDB_ENTITY=viano

# Prevent tokenizer fork warnings
export TOKENIZERS_PARALLELISM=false

# Run on GPU 0 only
export CUDA_VISIBLE_DEVICES=1

cd ~/trl/trl/experimental/gkd

# ------------------------
# Auth
# ------------------------
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
python -c "import wandb; wandb.login()"

# ------------------------
# Experimental knobs
# (edit here without touching python file)
# ------------------------
# Teacher: Llama-3.1-8B-Instruct (requires HF access to meta-llama)
export TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct
# Student: Llama-3.2-1B-Instruct (smaller, same family)
export STUDENT_MODEL=Qwen/Qwen2.5-0.5B-Instruct

export TRAIN_SAMPLES=2000
export EVAL_SAMPLES=256

export MAX_STEPS=800
export BATCH_SIZE=1
export GRAD_ACCUM=8
export LR=2e-6

# OPD loss mode
export OPD_MODE=softmax

# Algorithm 4 inner-loop knobs
export NUM_INNER_STEPS=10      # L  — gradient steps on Q per outer step
export REPLAY_BUFFER_SIZE=10   # k  — number of past batches kept in replay buffer

# OUTPUT_DIR is auto-generated from model names if not set;
# override here only if you need a custom path.
# export OUTPUT_DIR=my_custom_dir

# ------------------------
# Run experiment
# ------------------------
python -u opd_experiment.py 2>&1 | tee "run_llama_$(date +%Y%m%d_%H%M%S).log"
