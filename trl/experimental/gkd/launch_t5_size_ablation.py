#!/usr/bin/env python3
"""
Launch GKD vs OPD comparison across 3 Flan-T5 student sizes on GSM8K.

Matches Figure 1 (right) of the OPD paper: T5-Small/Base/Large students
against a fine-tuned T5-XL teacher.  Flan-T5 variants are used as drop-in
replacements (Flan-T5-XL already knows math reasoning).

GPU layout — same-size GKD and OPD are chained sequentially on one GPU so
that 4 GPUs cover all 6 runs with no idle time:

  GPU 0  →  gkd_small  then  opd_small  (sequential in one session)
  GPU 1  →  gkd_base   then  opd_base   (sequential in one session)
  GPU 2  →  gkd_large  (alone)
  GPU 3  →  opd_large  (alone)

GPU memory estimates (bfloat16, student + xl-teacher + optimiser states):
  flan-t5-small  (~77M)   + flan-t5-xl (~2.85B) ≈  8 GB
  flan-t5-base   (~248M)  + flan-t5-xl (~2.85B) ≈  9 GB
  flan-t5-large  (~783M)  + flan-t5-xl (~2.85B) ≈ 13 GB

All fit comfortably on 24 GB GPUs.  Small and base are chained sequentially
(not concurrent) so their memory is fully released between the two runs.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# Each entry: (session_name, [scripts to run sequentially])
SESSIONS = [
    ("t5_gpu0", [
        "run_gsm8k_gkd_S-flan-t5-small_T-flan-t5-xl.sh",
        "run_gsm8k_opd_S-flan-t5-small_T-flan-t5-xl.sh",
    ]),
    ("t5_gpu1", [
        "run_gsm8k_gkd_S-flan-t5-base_T-flan-t5-xl.sh",
        "run_gsm8k_opd_S-flan-t5-base_T-flan-t5-xl.sh",
    ]),
    ("gkd_t5_large", [
        "run_gsm8k_gkd_S-flan-t5-large_T-flan-t5-xl.sh",
    ]),
    ("opd_t5_large", [
        "run_gsm8k_opd_S-flan-t5-large_T-flan-t5-xl.sh",
    ]),
]


def session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name], capture_output=True
    ).returncode == 0


def launch(session, scripts):
    if session_exists(session):
        print(
            f"[SKIP]  Session '{session}' already exists — "
            f"kill it first with: tmux kill-session -t {session}"
        )
        return False

    # Chain scripts with && so the next one starts only after the previous succeeds
    run_cmds = " && ".join(f"bash {p}" for p in scripts)
    cmd = (
        f"conda activate opd && {run_cmds}; "
        f"echo; echo '=== All runs finished. Session staying open. ==='; exec bash"
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", choices=["gkd", "opd", "both"], default="both",
        help="Which algorithm(s) to include (default: both)",
    )
    args = parser.parse_args()

    # Validate all scripts exist first
    for _, scripts in SESSIONS:
        for script_name in scripts:
            path = SCRIPT_DIR / script_name
            if not path.exists():
                print(f"[ERROR] Script not found: {path}")
                sys.exit(1)

    launched = 0
    for session, script_names in SESSIONS:
        # Filter by --algo: keep a session only if it contains at least one matching script
        if args.algo != "both":
            script_names = [s for s in script_names if f"_{args.algo}_" in s]
            if not script_names:
                continue

        scripts = [SCRIPT_DIR / s for s in script_names]
        labels = " → ".join(Path(s).stem for s in script_names)
        if launch(session, scripts):
            gpu_line = f"GPU {script_names[0].split('DEVICES=')[0]}" if False else ""
            print(f"[OK]    Session '{session}': {labels}")
            launched += 1

    print()
    print(f"Launched {launched} session(s).")
    print()
    print("  GPU 0  →  gkd_small → opd_small  (session: t5_gpu0)")
    print("  GPU 1  →  gkd_base  → opd_base   (session: t5_gpu1)")
    print("  GPU 2  →  gkd_large              (session: gkd_t5_large)")
    print("  GPU 3  →  opd_large              (session: opd_t5_large)")
    print()
    print("Attach to a session with:  tmux attach -t <session>")
    print("List all sessions with:    tmux ls")


if __name__ == "__main__":
    main()
