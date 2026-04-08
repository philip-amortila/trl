#!/usr/bin/env python3
"""
Launch the correction-network (Algorithm 8) comparison experiments in 9 tmux sessions.

Three sessions per mode run on separate GPUs:

  corr_entropy_0     GPU 0  →  entropy_baseline  use_correction=False  trust_region=False
  corr_entropy_1     GPU 0  →  entropy_baseline  use_correction=True   trust_region=False
  corr_entropy_tr1   GPU 3  →  entropy_baseline  use_correction=True   trust_region=True

  corr_expectation_0     GPU 1  →  expectation  use_correction=False  trust_region=False
  corr_expectation_1     GPU 1  →  expectation  use_correction=True   trust_region=False
  corr_expectation_tr1   GPU 4  →  expectation  use_correction=True   trust_region=True

  corr_stochastic_0     GPU 2  →  stochastic  use_correction=False  trust_region=False
  corr_stochastic_1     GPU 2  →  stochastic  use_correction=True   trust_region=False
  corr_stochastic_tr1   GPU 5  →  stochastic  use_correction=True   trust_region=True

Teacher : Qwen/Qwen2-1.5B-Instruct
Student : Qwen/Qwen2-0.5B-Instruct
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# (session_corr0, session_corr1, session_corrtr1,
#  script_corr0,  script_corr1,  script_corrtr1)
SESSIONS = [
    (
        "corr_entropy_0",
        "corr_entropy_1",
        "corr_entropy_tr1",
        "run_gsm8k_opd_entropy_T-Qwen2-1.5B-Instruct_corr0.sh",
        "run_gsm8k_opd_entropy_T-Qwen2-1.5B-Instruct_corr1.sh",
        "run_gsm8k_opd_entropy_T-Qwen2-1.5B-Instruct_corrtr1.sh",
    ),
    (
        "corr_expectation_0",
        "corr_expectation_1",
        "corr_expectation_tr1",
        "run_gsm8k_opd_expectation_T-Qwen2-1.5B-Instruct_corr0.sh",
        "run_gsm8k_opd_expectation_T-Qwen2-1.5B-Instruct_corr1.sh",
        "run_gsm8k_opd_expectation_T-Qwen2-1.5B-Instruct_corrtr1.sh",
    ),
    (
        "corr_stochastic_0",
        "corr_stochastic_1",
        "corr_stochastic_tr1",
        "run_gsm8k_opd_stochastic_T-Qwen2-1.5B-Instruct_corr0.sh",
        "run_gsm8k_opd_stochastic_T-Qwen2-1.5B-Instruct_corr1.sh",
        "run_gsm8k_opd_stochastic_T-Qwen2-1.5B-Instruct_corrtr1.sh",
    ),
]


def session_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
    )
    return result.returncode == 0


def launch(session: str, script_path: Path) -> bool:
    """Create a new tmux session running script_path. Returns True if launched."""
    if session_exists(session):
        print(
            f"[SKIP]  Session '{session}' already exists — "
            f"kill it first with: tmux kill-session -t {session}"
        )
        return False

    cmd = (
        f"conda activate opd && bash {script_path}; "
        f"echo; echo '=== Run finished. Session staying open. ==='; exec bash"
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
    for sess0, sess1, sesstr1, script_corr0, script_corr1, script_corrtr1 in SESSIONS:
        paths = {
            sess0: SCRIPT_DIR / script_corr0,
            sess1: SCRIPT_DIR / script_corr1,
            sesstr1: SCRIPT_DIR / script_corrtr1,
        }

        for path in paths.values():
            if not path.exists():
                print(f"[ERROR] Script not found: {path}")
                sys.exit(1)

        for session, path in paths.items():
            if launch(session, path):
                print(f"[OK]    Session '{session}' started → {path.name}")

    print()
    print("Attach to a session with:  tmux attach -t <session>")
    print("List all sessions with:    tmux ls")


if __name__ == "__main__":
    main()
