#!/usr/bin/env python3
"""
Launch the four 14B-teacher experiments in separate tmux sessions.
Each session stays open (drops into bash) after the script finishes.

Sessions and scripts:
  opd5  →  run_gsm8k_gkd_T-Qwen2.5-14B-Instruct.sh
  opd6  →  run_gsm8k_opd_softmax_T-Qwen2.5-14B-Instruct.sh
  opd7  →  run_gsm8k_opd_expectation_T-Qwen2.5-14B-Instruct.sh
  opd8  →  run_gsm8k_opd_entropy_T-Qwen2.5-14B-Instruct.sh
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

SESSIONS = [
    ("opd5", "run_gsm8k_gkd_T-Qwen2.5-14B-Instruct.sh"),
    ("opd6", "run_gsm8k_opd_softmax_T-Qwen2.5-14B-Instruct.sh"),
    ("opd7", "run_gsm8k_opd_expectation_T-Qwen2.5-14B-Instruct.sh"),
    ("opd8", "run_gsm8k_opd_entropy_T-Qwen2.5-14B-Instruct.sh"),
]


def session_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
    )
    return result.returncode == 0


def main() -> None:
    for session, script in SESSIONS:
        script_path = SCRIPT_DIR / script

        if not script_path.exists():
            print(f"[ERROR] Script not found: {script_path}")
            sys.exit(1)

        if session_exists(session):
            print(f"[SKIP]  Session '{session}' already exists — kill it first with: tmux kill-session -t {session}")
            continue

        # Activate the conda environment, run the script, then drop into bash so the session stays open.
        cmd = f"conda activate opd && bash {script_path}; echo; echo '=== Experiment finished. Session staying open. ==='; exec bash"

        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, "-x", "220", "-y", "50"],
            check=True,
        )
        subprocess.run(
            ["tmux", "send-keys", "-t", session, cmd, "Enter"],
            check=True,
        )
        print(f"[OK]    Session '{session}' started → {script}")

    print()
    print("Attach to a session with:  tmux attach -t <session>")
    print("List all sessions with:    tmux ls")


if __name__ == "__main__":
    main()
