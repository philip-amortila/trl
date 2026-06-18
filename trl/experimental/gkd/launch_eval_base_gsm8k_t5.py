#!/usr/bin/env python3
"""
Evaluate the 3 Flan-T5 student models on GSM8K before any distillation training.

Models:
  google/flan-t5-small  → GPU 0
  google/flan-t5-base   → GPU 1
  google/flan-t5-large  → GPU 2

Results are saved as eval_gsm8k_t5.json inside a per-model directory
(e.g. eval_google_flan-t5-small/) under the script directory.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

if os.environ.get("CONDA_DEFAULT_ENV") != "opd":
    sys.exit(subprocess.run(["conda", "run", "-n", "opd", "python"] + sys.argv).returncode)

SCRIPT_DIR = Path(__file__).parent.resolve()
EVAL_T5 = SCRIPT_DIR / "eval_gsm8k_t5.py"

MODELS = [
    ("eval_base_t5_small", 0, "google/flan-t5-small"),
    ("eval_base_t5_base",  1, "google/flan-t5-base"),
    ("eval_base_t5_large", 2, "google/flan-t5-large"),
]


def session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={name}"], capture_output=True
    ).returncode == 0


def launch(session: str, gpu: int, model_id: str) -> bool:
    if session_exists(session):
        print(
            f"[SKIP]  Session '{session}' already exists — "
            f"kill it first with: tmux kill-session -t {session}"
        )
        return False

    out_dir = SCRIPT_DIR / ("eval_" + model_id.replace("/", "_"))
    log = out_dir / "eval_gsm8k_t5_base.log"

    script_body = (
        f"#!/bin/bash\n"
        f"source $(conda info --base)/etc/profile.d/conda.sh\n"
        f"conda activate opd\n"
        f"cd {SCRIPT_DIR}\n"
        f"mkdir -p {out_dir}\n"
        f"echo '=== Evaluating {model_id} on GSM8K ==='\n"
        f"CUDA_VISIBLE_DEVICES={gpu} python -u {EVAL_T5}"
        f" --model_dir {model_id}"
        f" --output_dir {out_dir}"
        f" --teacher_model_dir google/flan-t5-xl"
        f" 2>&1 | tee {log}\n"
        f"echo\n"
        f"echo '=== Done. Session staying open. ==='\n"
        f"exec bash\n"
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", prefix=f"eval_{session}_", delete=False
    ) as f:
        f.write(script_body)
        script_path = f.name

    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-x", "220", "-y", "50"],
        check=True,
    )
    subprocess.run(
        ["tmux", "send-keys", "-t", f"{session}:0.0", f"bash {script_path}", "Enter"],
        check=True,
    )
    return True


def main() -> None:
    if not EVAL_T5.exists():
        print(f"[ERROR] Eval script not found: {EVAL_T5}")
        sys.exit(1)

    print("Flan-T5 student baselines to evaluate on GSM8K:")
    for session, gpu, model_id in MODELS:
        print(f"  GPU {gpu}  {session:<24}  {model_id}")
    print()

    launched = 0
    for session, gpu, model_id in MODELS:
        if launch(session, gpu, model_id):
            print(f"[OK]    Session '{session}' started on GPU {gpu} → {model_id}")
            launched += 1

    print()
    print(f"Launched {launched}/{len(MODELS)} sessions.")
    print("Attach to a session with:  tmux attach -t <session>")
    print("List all sessions with:    tmux ls")
    print()
    print("Results will be saved to:")
    for _, _, model_id in MODELS:
        out_dir = SCRIPT_DIR / ("eval_" + model_id.replace("/", "_"))
        print(f"  {out_dir}/eval_gsm8k_t5.json")


if __name__ == "__main__":
    main()
