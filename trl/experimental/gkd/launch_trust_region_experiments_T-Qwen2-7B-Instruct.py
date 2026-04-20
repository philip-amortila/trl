#!/usr/bin/env python3
"""
Launch the trust-region comparison experiments in 8 tmux sessions.

Each GPU hosts two sessions that run simultaneously (tr0 and tr1 in parallel):

  tr_gkd_0         GPU 0  →  gkd          trust_region=False  (tr is N/A for GKD)
  tr_gkd_1         GPU 0  →  gkd          trust_region=True   (tr is N/A for GKD)
  tr_entropy_0     GPU 1  →  entropy      trust_region=False
  tr_entropy_1     GPU 1  →  entropy      trust_region=True
  tr_expectation_0 GPU 2  →  expectation  trust_region=False
  tr_expectation_1 GPU 2  →  expectation  trust_region=True
  tr_stochastic_0  GPU 3  →  stochastic   trust_region=False
  tr_stochastic_1  GPU 3  →  stochastic   trust_region=True

Teacher : Qwen/Qwen2-7B-Instruct
Student : Qwen/Qwen2-0.5B-Instruct
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# (session_tr0, session_tr1, script_tr0, script_tr1)
SESSIONS = [
    (
        "tr_gkd_0",
        "tr_gkd_1",
        "run_gsm8k_gkd_T-Qwen2-7B-Instruct_tr0.sh",
        "run_gsm8k_gkd_T-Qwen2-7B-Instruct_tr1.sh",
    ),
    (
        "tr_entropy_0",
        "tr_entropy_1",
        "run_gsm8k_opd_entropy_T-Qwen2-7B-Instruct_tr0.sh",
        "run_gsm8k_opd_entropy_T-Qwen2-7B-Instruct_tr1.sh",
    ),
    (
        "tr_expectation_0",
        "tr_expectation_1",
        "run_gsm8k_opd_expectation_T-Qwen2-7B-Instruct_tr0.sh",
        "run_gsm8k_opd_expectation_T-Qwen2-7B-Instruct_tr1.sh",
    ),
    (
        "tr_stochastic_0",
        "tr_stochastic_1",
        "run_gsm8k_opd_stochastic_T-Qwen2-7B-Instruct_tr0.sh",
        "run_gsm8k_opd_stochastic_T-Qwen2-7B-Instruct_tr1.sh",
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
    for sess0, sess1, script_tr0, script_tr1 in SESSIONS:
        path_tr0 = SCRIPT_DIR / script_tr0
        path_tr1 = SCRIPT_DIR / script_tr1

        for path in (path_tr0, path_tr1):
            if not path.exists():
                print(f"[ERROR] Script not found: {path}")
                sys.exit(1)

        # Launch both sessions simultaneously (same GPU, different tmux sessions).
        if launch(sess0, path_tr0):
            print(f"[OK]    Session '{sess0}' started → {script_tr0}")
        if launch(sess1, path_tr1):
            print(f"[OK]    Session '{sess1}' started → {script_tr1}")

    print()
    print("Attach to a session with:  tmux attach -t <session>")
    print("List all sessions with:    tmux ls")


if __name__ == "__main__":
    main()
