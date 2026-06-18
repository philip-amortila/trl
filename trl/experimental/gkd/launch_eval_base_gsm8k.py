#!/usr/bin/env python3
"""
Evaluate the four base (instruct) models on GSM8K before any distillation training.

Models:
  Qwen/Qwen2-0.5B-Instruct   → GPU 1
  Qwen/Qwen2-1.5B-Instruct   → GPU 2
  Qwen/Qwen2.5-3B-Instruct   → GPU 3
  Qwen/Qwen2-7B-Instruct     → GPU 0

Results are saved as eval_gsm8k.json inside a per-model directory
(e.g. eval_Qwen_Qwen2-0.5B-Instruct/) under the script directory.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Re-launch inside the opd conda environment if not already active
if os.environ.get("CONDA_DEFAULT_ENV") != "opd":
    sys.exit(subprocess.run(["conda", "run", "-n", "opd", "python"] + sys.argv).returncode)

SCRIPT_DIR = Path(__file__).parent.resolve()
EVAL_GSM8K = SCRIPT_DIR / "eval_gsm8k.py"

MODELS = [
    ("eval_base_gsm8k_05B",  1, "Qwen/Qwen2-0.5B-Instruct"),
    ("eval_base_gsm8k_15B",  2, "Qwen/Qwen2-1.5B-Instruct"),
    ("eval_base_gsm8k_3B",   3, "Qwen/Qwen2.5-3B-Instruct"),
    ("eval_base_gsm8k_7B",   0, "Qwen/Qwen2-7B-Instruct"),
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
    log = out_dir / "eval_gsm8k_base.log"

    script_body = (
        f"#!/bin/bash\n"
        f"source $(conda info --base)/etc/profile.d/conda.sh\n"
        f"conda activate opd\n"
        f"mkdir -p ~/.cache/huggingface\n"
        f"echo -n 'YOUR_HF_TOKEN' > ~/.cache/huggingface/token\n"
        f"export HF_TOKEN='YOUR_HF_TOKEN'\n"
        f"python -c \"from huggingface_hub import login; login(token='${{HF_TOKEN}}')\"\n"
        f"cd {SCRIPT_DIR}\n"
        f"mkdir -p {out_dir}\n"
        f"echo '=== Evaluating {model_id} ==='\n"
        f"CUDA_VISIBLE_DEVICES={gpu} python -u {EVAL_GSM8K}"
        f" --model_dir {model_id}"
        f" --output_dir {out_dir}"
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
    if not EVAL_GSM8K.exists():
        print(f"[ERROR] Eval script not found: {EVAL_GSM8K}")
        sys.exit(1)

    print("Base models to evaluate on GSM8K:")
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
        print(f"  {out_dir}/eval_gsm8k.json")


if __name__ == "__main__":
    main()
