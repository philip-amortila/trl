#!/usr/bin/env python3
"""
Launch GKD vs OPD-expectation+TR comparison across 3 student sizes.

  opd_s05  GPU 0  →  opd expectation  trust_region=True  student=Qwen2-0.5B
  opd_s15  GPU 1  →  opd expectation  trust_region=True  student=Qwen2-1.5B
  opd_s7b  GPU 2  →  opd expectation  trust_region=True  student=Qwen2-7B   (LoRA)
  gkd_s05  GPU 3  →  gkd              student=Qwen2-0.5B
  gkd_s15  GPU 4  →  gkd              student=Qwen2-1.5B
  gkd_s7b  GPU 5  →  gkd              student=Qwen2-7B   (LoRA)

Teacher : Qwen/Qwen2-7B-Instruct
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# (session, script)
SESSIONS = [
    (
        "opd_s05",
        "run_gsm8k_opd_expectation_S-Qwen2-0.5B-Instruct_T-Qwen2-7B-Instruct_tr1.sh",
    ),
    (
        "opd_s15",
        "run_gsm8k_opd_expectation_S-Qwen2-1.5B-Instruct_T-Qwen2-7B-Instruct_tr1.sh",
    ),
    (
        "opd_s7b",
        "run_gsm8k_opd_expectation_S-Qwen2-7B-Instruct_T-Qwen2-7B-Instruct_tr1.sh",
    ),
    (
        "gkd_s05",
        "run_gsm8k_gkd_S-Qwen2-0.5B-Instruct_T-Qwen2-7B-Instruct.sh",
    ),
    (
        "gkd_s15",
        "run_gsm8k_gkd_S-Qwen2-1.5B-Instruct_T-Qwen2-7B-Instruct.sh",
    ),
    (
        "gkd_s7b",
        "run_gsm8k_gkd_S-Qwen2-7B-Instruct_T-Qwen2-7B-Instruct.sh",
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
    for session, script_name in SESSIONS:
        path = SCRIPT_DIR / script_name
        if not path.exists():
            print(f"[ERROR] Script not found: {path}")
            sys.exit(1)
        if launch(session, path):
            print(f"[OK]    Session '{session}' started → {script_name}")

    print()
    print("Attach to a session with:  tmux attach -t <session>")
    print("List all sessions with:    tmux ls")


if __name__ == "__main__":
    main()
