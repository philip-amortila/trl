#!/usr/bin/env python3
"""
Evaluate the 8 models produced by launch_trust_region_experiments.py.

Auto-discovers the most recent output directory for each of the 8 runs
(4 algorithms × 2 trust_region settings) and launches one tmux session per
model.  Two sessions share the same GPU so both evaluations run simultaneously.

GPU assignment:
  GPU 0  →  eval_gkd_0        gkd  tr=False
             eval_gkd_1        gkd  tr=True  (same behaviour – GKD has no TR)
  GPU 1  →  eval_entropy_0    entropy_baseline  tr=False
             eval_entropy_1    entropy_baseline  tr=True
  GPU 2  →  eval_expectation_0  expectation  tr=False
             eval_expectation_1  expectation  tr=True
  GPU 3  →  eval_stochastic_0   stochastic   tr=False
             eval_stochastic_1   stochastic   tr=True

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


def find_two_latest(pattern: str) -> tuple[Path, Path]:
    """Return the two most recently modified directories matching a glob pattern."""
    matches = sorted(SCRIPT_DIR.glob(pattern), key=lambda d: d.stat().st_mtime)
    if len(matches) < 2:
        raise FileNotFoundError(
            f"Need at least 2 directories matching: {SCRIPT_DIR / pattern}  (found {len(matches)})"
        )
    return matches[-2], matches[-1]


def resolve_models() -> list[tuple[str, int, Path]]:
    """
    Return a list of (session_name, gpu_index, model_dir) for all 8 evals.
    Directories are discovered at launch time so the script works regardless
    of which exact timestamp was assigned during training.
    """
    # gkd: both runs have the same naming (no _tr infix); take the 2 latest.
    gkd_tr0, gkd_tr1 = find_two_latest(
        "gkd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_*"
    )

    # OPD modes: distinguish tr=False (no _tr_ infix) from tr=True (_tr_ infix).
    entropy_tr0  = find_latest("opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_entropy_baseline_L*_buf*")
    entropy_tr1  = find_latest("opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_entropy_baseline_tr_L*_buf*")
    expect_tr0   = find_latest("opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_expectation_L*_buf*")
    expect_tr1   = find_latest("opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_expectation_tr_L*_buf*")
    stoch_tr0    = find_latest("opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_stochastic_L*_buf*")
    stoch_tr1    = find_latest("opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_stochastic_tr_L*_buf*")

    return [
        ("eval_gkd_0",         0, gkd_tr0),
        ("eval_gkd_1",         0, gkd_tr1),
        ("eval_entropy_0",     1, entropy_tr0),
        ("eval_entropy_1",     1, entropy_tr1),
        ("eval_expectation_0", 2, expect_tr0),
        ("eval_expectation_1", 2, expect_tr1),
        ("eval_stochastic_0",  3, stoch_tr0),
        ("eval_stochastic_1",  3, stoch_tr1),
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
        print(f"  GPU {gpu}  {session:<22}  {model_dir.name}")
    print()

    for session, gpu, model_dir in jobs:
        if launch(session, gpu, model_dir):
            print(f"[OK]    Session '{session}' started on GPU {gpu} → {model_dir.name}")

    print()
    print("Attach to a session with:  tmux attach -t <session>")
    print("List all sessions with:    tmux ls")


if __name__ == "__main__":
    main()
