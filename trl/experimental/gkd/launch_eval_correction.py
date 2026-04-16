#!/usr/bin/env python3
"""
Evaluate the 9 models produced by launch_correction_experiments.py.

Auto-discovers the most recent output directory for each of the 9 runs
(3 algorithms × 3 correction/trust-region settings) and launches one tmux
session per model.

GPU assignment mirrors the training layout:
  GPU 0  →  eval_entropy_corr0       entropy_baseline  correction=False  tr=False
             eval_entropy_corr1       entropy_baseline  correction=True   tr=False
  GPU 1  →  eval_expectation_corr0   expectation       correction=False  tr=False
             eval_expectation_corr1   expectation       correction=True   tr=False
  GPU 2  →  eval_stochastic_corr0    stochastic        correction=False  tr=False
             eval_stochastic_corr1    stochastic        correction=True   tr=False
  GPU 3  →  eval_entropy_corrtr1     entropy_baseline  correction=True   tr=True
             eval_expectation_corrtr1 expectation       correction=True   tr=True
             eval_stochastic_corrtr1  stochastic        correction=True   tr=True

Teacher : Qwen/Qwen2-1.5B-Instruct
Student : Qwen/Qwen2-0.5B-Instruct
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
EVAL_SCRIPT = SCRIPT_DIR / "eval_gsm8k.py"


def find_latest(pattern: str) -> Path:
    """Return the most recently modified directory matching a glob pattern."""
    matches = sorted(SCRIPT_DIR.glob(pattern), key=lambda d: d.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No directory found matching: {SCRIPT_DIR / pattern}")
    return matches[-1]


def resolve_models() -> list[tuple[str, int, Path]]:
    """
    Return a list of (session_name, gpu_index, model_dir) for all 9 evals.
    Directories are discovered at launch time so the script works regardless
    of which exact timestamp was assigned during training.
    """
    base = "opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct"

    # corr0: no correction, no trust-region  → no _corr / _tr infix
    entropy_corr0    = find_latest(f"{base}_entropy_baseline_L*_buf*")
    expectation_corr0 = find_latest(f"{base}_expectation_L*_buf*")
    stochastic_corr0  = find_latest(f"{base}_stochastic_L*_buf*")

    # corr1: correction=True, trust_region=False  → _corr infix, no _tr
    entropy_corr1    = find_latest(f"{base}_entropy_baseline_corr_L*_buf*")
    expectation_corr1 = find_latest(f"{base}_expectation_corr_L*_buf*")
    stochastic_corr1  = find_latest(f"{base}_stochastic_corr_L*_buf*")

    # corrtr1: correction=True, trust_region=True  → _tr_corr infix
    entropy_corrtr1    = find_latest(f"{base}_entropy_baseline_tr_corr_L*_buf*")
    expectation_corrtr1 = find_latest(f"{base}_expectation_tr_corr_L*_buf*")
    stochastic_corrtr1  = find_latest(f"{base}_stochastic_tr_corr_L*_buf*")

    return [
        ("eval_entropy_corr0",      0, entropy_corr0),
        ("eval_entropy_corr1",      0, entropy_corr1),
        ("eval_expectation_corr0",  1, expectation_corr0),
        ("eval_expectation_corr1",  1, expectation_corr1),
        ("eval_stochastic_corr0",   2, stochastic_corr0),
        ("eval_stochastic_corr1",   2, stochastic_corr1),
        ("eval_entropy_corrtr1",    3, entropy_corrtr1),
        ("eval_expectation_corrtr1",3, expectation_corrtr1),
        ("eval_stochastic_corrtr1", 3, stochastic_corrtr1),
    ]


def session_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
    )
    return result.returncode == 0


def launch(session: str, gpu: int, model_dir: Path) -> bool:
    """Create a tmux session that evaluates model_dir on the given GPU."""
    if session_exists(session):
        print(
            f"[SKIP]  Session '{session}' already exists — "
            f"kill it first with: tmux kill-session -t {session}"
        )
        return False

    log_path = model_dir / "eval_gsm8k_launch.log"
    cmd = (
        f"conda activate opd && "
        f"cd {SCRIPT_DIR} && "
        f"CUDA_VISIBLE_DEVICES={gpu} python -u {EVAL_SCRIPT} --model_dir {model_dir} "
        f"2>&1 | tee {log_path}; "
        f"echo; echo '=== Eval finished. Session staying open. ==='; exec bash"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-x", "220", "-y", "50"],
        check=True,
    )
    subprocess.run(
        ["tmux", "send-keys", "-t", session, cmd, "Enter"],
        check=True,
    )
    return True


def main() -> None:
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
        print(f"  GPU {gpu}  {session:<26}  {model_dir.name}")
    print()

    for session, gpu, model_dir in jobs:
        if launch(session, gpu, model_dir):
            print(f"[OK]    Session '{session}' started on GPU {gpu} → {model_dir.name}")

    print()
    print("Attach to a session with:  tmux attach -t <session>")
    print("List all sessions with:    tmux ls")


if __name__ == "__main__":
    main()
