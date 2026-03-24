#!/usr/bin/env bash
#SBATCH -p songmei
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -t 8:00:00
#SBATCH -J gsm8k_opd_expectation_llama
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -e

export WANDB_ENTITY=viano
export TOKENIZERS_PARALLELISM=false

cd ~/trl/trl/experimental/gkd

export TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct
export STUDENT_MODEL=Qwen/Qwen2.5-0.5B-Instruct

export TRAIN_SAMPLES=2000
export EVAL_SAMPLES=256

export MAX_STEPS=800
export BATCH_SIZE=1
export GRAD_ACCUM=8
export LR=2e-6

export OPD_MODE=expectation
export NUM_INNER_STEPS=10
export REPLAY_BUFFER_SIZE=10

python -u opd_experiment.py
