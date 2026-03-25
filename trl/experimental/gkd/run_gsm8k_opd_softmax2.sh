#!/usr/bin/env bash
set -e

export WANDB_ENTITY=viano

# Prevent tokenizer fork warnings
export TOKENIZERS_PARALLELISM=false

# Run on GPU 0 only
export CUDA_VISIBLE_DEVICES=1

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

# OPD loss mode: "softmax"  (Algorithm 4, softmax mode)
# Maximises E_{pi_E}[Q_theta] - E_{pi_theta}[Q_theta] analytically,
# treating student logits as Q_theta and detaching the current policy weights.
export OPD_MODE=softmax

# Algorithm 4 inner-loop knobs
export NUM_INNER_STEPS=50      # L  — gradient steps on Q per outer step
export REPLAY_BUFFER_SIZE=10   # k  — number of past batches kept in replay buffer

export OUTPUT_DIR=opd_gsm8k_out_${OPD_MODE}_L${NUM_INNER_STEPS}_buf${REPLAY_BUFFER_SIZE}_$(date +%Y%m%d_%H%M%S)

# ------------------------
# Run experiment
# ------------------------
mkdir -p "$OUTPUT_DIR"
python -u opd_experiment.py 2>&1 | tee "${OUTPUT_DIR}/run.log"
