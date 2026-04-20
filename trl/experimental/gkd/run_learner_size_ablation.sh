#!/usr/bin/env bash
set -e

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd ~/trl/trl/experimental/gkd

export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
python -c "import wandb; wandb.login()"

for STUDENT in \
    "Qwen/Qwen2-7B-Instruct" \
    "Qwen/Qwen2-1.5B-Instruct" \
    "Qwen/Qwen2-0.5B-Instruct"
do
    echo ""
    echo "=================================================="
    echo "Starting: STUDENT_MODEL=${STUDENT}"
    echo "=================================================="
    STUDENT_MODEL="${STUDENT}" python -u run_shared_teacher_experiment.py \
        2>&1 | tee "learner_ablation_$(echo ${STUDENT} | tr '/' '_')_$(date +%Y%m%d_%H%M%S).log"
    echo "Finished: STUDENT_MODEL=${STUDENT}"
done

echo ""
echo "All three student sizes complete."
