#!/usr/bin/env python3
"""
Discover the 6 models produced by run_learner_size_ablation.sh and evaluate
each on GSM8K, MATH500, and SVAMP using eval_math_benchmarks.py.

One tmux session per model, one GPU each (GPUs 0-5).

  GPU 0  →  eval_abl_gkd_7b     GKD   Qwen2-7B student
  GPU 1  →  eval_abl_opd_7b     OPD   Qwen2-7B student
  GPU 2  →  eval_abl_gkd_1b5    GKD   Qwen2-1.5B student
  GPU 3  →  eval_abl_opd_1b5    OPD   Qwen2-1.5B student
  GPU 4  →  eval_abl_gkd_0b5    GKD   Qwen2-0.5B student
  GPU 5  →  eval_abl_opd_0b5    OPD   Qwen2-0.5B student

Teacher is assumed to be Qwen2-57B-A14B-Instruct (set TEACHER env var to override).
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent.resolve()
EVAL_SCRIPT = SCRIPT_DIR / "eval_math_benchmarks.py"

TEACHER = "Qwen2-57B-A14B-Instruct"  # short name used in directory patterns

STUDENTS = [
    ("7B",   "Qwen2-7B-Instruct",   0, 1),   # (label, short_name, gkd_gpu, opd_gpu)
    ("1.5B", "Qwen2-1.5B-Instruct", 2, 3),
    ("0.5B", "Qwen2-0.5B-Instruct", 4, 5),
]


def find_latest(pattern: str) -> Path:
    matches = sorted(SCRIPT_DIR.glob(pattern), key=lambda d: d.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No directory matching: {SCRIPT_DIR / pattern}")
    return matches[-1]


def resolve_models() -> list[tuple[str, int, Path]]:
    jobs = []
    for label, student, gkd_gpu, opd_gpu in STUDENTS:
        gkd_dir = find_latest(f"gkd_gsm8k_S-{student}_T-{TEACHER}_*")
        opd_dir = find_latest(f"opd_gsm8k_S-{student}_T-{TEACHER}_expectation_tr_*")
        jobs.append((f"eval_abl_gkd_{label.replace('.', 'b')}", gkd_gpu, gkd_dir))
        jobs.append((f"eval_abl_opd_{label.replace('.', 'b')}", opd_gpu, opd_dir))
    return jobs


def session_exists(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name],
                          capture_output=True).returncode == 0


def launch(session: str, gpu: int, model_dir: Path) -> bool:
    if session_exists(session):
        print(f"[SKIP]  '{session}' already exists — kill first: tmux kill-session -t {session}")
        return False

    log = model_dir / "eval_benchmarks.log"
    cmd = (
        f"conda activate opd && "
        f"cd {SCRIPT_DIR} && "
        f"CUDA_VISIBLE_DEVICES={gpu} python -u {EVAL_SCRIPT} "
        f"--model_dir {model_dir} "
        f"2>&1 | tee {log}; "
        f"echo; echo '=== Eval done ==='; exec bash"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-x", "220", "-y", "50"], check=True)
    subprocess.run(["tmux", "send-keys", "-t", session, cmd, "Enter"], check=True)
    return True


def main():
    if not EVAL_SCRIPT.exists():
        print(f"[ERROR] Eval script not found: {EVAL_SCRIPT}")
        sys.exit(1)

    try:
        jobs = resolve_models()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("Discovered model directories:")
    for session, gpu, model_dir in jobs:
        print(f"  GPU {gpu}  {session:<25}  {model_dir.name}")
    print()

    for session, gpu, model_dir in jobs:
        if launch(session, gpu, model_dir):
            print(f"[OK]  '{session}' started on GPU {gpu} → {model_dir.name}")

    print()
    print("Attach with:   tmux attach -t <session>")
    print("List all:      tmux ls")


if __name__ == "__main__":
    main()
