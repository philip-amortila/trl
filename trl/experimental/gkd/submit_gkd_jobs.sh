#!/usr/bin/env bash
# submit_gkd_jobs.sh
#
# Submits one independent SLURM job per student model on Clariden (CSCS).
# Each job loads its own copy of the teacher on its own GH200 GPU.
#
# Clariden specifics (https://docs.cscs.ch/clusters/clariden/,
#                     https://docs.cscs.ch/running/slurm/):
#   - GH200 nodes, 4 GPUs per node; allocate with --gpus-per-task (not --gres)
#   - normal partition: 12-hour wall-clock limit
#   - --account is mandatory; fill in ACCOUNT below or set it via env
#
# Cluster knobs (override via environment before calling this script):
#   ACCOUNT     SLURM project account      (REQUIRED, e.g. g123)
#   PARTITION   SLURM partition            (default: normal)
#   CPUS        CPUs per task              (default: 72)
#   MEM         Host memory per job        (default: 120G)
#   TIME        Wall-clock time limit      (default: 12:00:00)
#   OUTPUT_DIR  Root dir for all outputs   (default: gkd_qwen35_experiments)
#   CONDA_ENV   Conda environment to activate (default: ovi)
#
# Training knobs forwarded to run_gkd_qwen35.py:
#   TEACHER_MODEL  MAX_STEPS  SAVE_STEPS  BATCH_SIZE  GRAD_ACCUM  LR  BETA
#
# Usage:
#   bash submit_gkd_jobs.sh
#   PARTITION=debug TIME=01:00:00 bash submit_gkd_jobs.sh
#   CONDA_ENV=other_env bash submit_gkd_jobs.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Cluster defaults — Clariden / Alps
# ---------------------------------------------------------------------------
ACCOUNT="${ACCOUNT:-a0136}"
if [[ -z "${ACCOUNT}" ]]; then
    echo "ERROR: ACCOUNT is required. Set it to your CSCS project group, e.g.:"
    echo "  ACCOUNT=g123 bash submit_gkd_jobs.sh"
    exit 1
fi

CONDA_ENV="${CONDA_ENV:-ovi}"
PARTITION="${PARTITION:-normal}"
CPUS="${CPUS:-72}"          # GH200 node has ~72 Grace CPU cores
MEM="${MEM:-120G}"          # node total is ~450 GB; leave room for the OS and other jobs
TIME="${TIME:-12:00:00}"    # normal partition wall-clock limit on Clariden

# ---------------------------------------------------------------------------
# Experiment defaults (passed through to the Python script)
# ---------------------------------------------------------------------------
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-9B}"
MAX_STEPS="${MAX_STEPS:-600}"
SAVE_STEPS="${SAVE_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-6}"
BETA="${BETA:-0.5}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/gkd_qwen35_experiments}"
LOG_DIR="${OUTPUT_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Students — one job each
# ---------------------------------------------------------------------------
STUDENTS=(
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-2B"
    "Qwen/Qwen3.5-0.8B"
)

echo "Submitting ${#STUDENTS[@]} GKD jobs on Clariden"
echo "  Account    : ${ACCOUNT}"
echo "  Partition  : ${PARTITION}  TIME=${TIME}"
echo "  Conda env  : ${CONDA_ENV}"
echo "  Per job    : 1 node, 1 task, 1 GH200 GPU, ${CPUS} CPUs, ${MEM} RAM"
echo "  Teacher    : ${TEACHER_MODEL}"
echo "  Max steps  : ${MAX_STEPS}  (checkpoint every ${SAVE_STEPS})"
echo "  Output root: ${OUTPUT_DIR}"
echo ""

for STUDENT in "${STUDENTS[@]}"; do
    SHORT="${STUDENT##*/}"   # e.g. Qwen3.5-4B

    JOB_ID=$(sbatch --parsable \
        --account="${ACCOUNT}" \
        --job-name="gkd_${SHORT}" \
        --partition="${PARTITION}" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gpus-per-task=1 \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --output="${LOG_DIR}/${SHORT}_%j.out" \
        --error="${LOG_DIR}/${SHORT}_%j.err" \
        <<JOB_SCRIPT
#!/usr/bin/env bash
set -euo pipefail

# ---- activate conda environment ----
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ---- forward experiment settings ----
export STUDENT_MODEL="${STUDENT}"
export TEACHER_MODEL="${TEACHER_MODEL}"
export OUTPUT_DIR="${OUTPUT_DIR}"
export MAX_STEPS="${MAX_STEPS}"
export SAVE_STEPS="${SAVE_STEPS}"
export BATCH_SIZE="${BATCH_SIZE}"
export GRAD_ACCUM="${GRAD_ACCUM}"
export LR="${LR}"
export BETA="${BETA}"

# ---- HF token ----
export HUGGING_FACE_HUB_TOKEN="<YOUR_HF_TOKEN>"
export HF_TOKEN="<YOUR_HF_TOKEN>"

echo "=== Job info ==="
echo "SLURM_JOB_ID        : \${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST  : \${SLURM_JOB_NODELIST:-<local>}"
echo "CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "Student             : ${STUDENT}"
echo "Teacher             : ${TEACHER_MODEL}"
echo "================"

cd "${SCRIPT_DIR}"
srun python run_gkd_qwen35.py
JOB_SCRIPT
    )

    echo "  Submitted  ${STUDENT}  →  job ${JOB_ID}"
    echo "             logs: ${LOG_DIR}/${SHORT}_${JOB_ID}.{out,err}"
done

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/<student>_<jobid>.out"
