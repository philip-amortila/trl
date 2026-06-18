#!/usr/bin/env python3
"""
Evaluate all checkpoints of the 6 T5 models from the T5 size ablation
(3 sizes × 2 algorithms = 6 runs trained with launch_t5_size_ablation.py).

For each model the script evaluates every checkpoint-N/ subdirectory found
at launch time, then the final root model (if present), all sequentially
inside one tmux session.

Uses eval_gsm8k_t5.py — the flexible T5 evaluation protocol with strict
and flexible answer parsing (counterpart of eval_gsm8k.py for seq2seq models).

Students: flan-t5-small, flan-t5-base, flan-t5-large
Teacher:  flan-t5-xl
Algorithms: GKD, OPD (expectation + trust-region)

GPU layout (mirrors launch_t5_size_ablation.py training layout):
  GPU 0  →  eval_gkd_t5_small  eval_opd_t5_small  (concurrent; models are tiny)
  GPU 1  →  eval_gkd_t5_base   eval_opd_t5_base   (concurrent; models are small)
  GPU 2  →  eval_gkd_t5_large
  GPU 3  →  eval_opd_t5_large
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
EVAL_T5 = SCRIPT_DIR / "eval_gsm8k_t5.py"

SPECS = [
    ("eval_gkd_t5_small", 0, "gkd_gsm8k_S-flan-t5-small_T-flan-t5-xl_*"),
    ("eval_gkd_t5_base",  1, "gkd_gsm8k_S-flan-t5-base_T-flan-t5-xl_*"),
    ("eval_gkd_t5_large", 2, "gkd_gsm8k_S-flan-t5-large_T-flan-t5-xl_*"),
    ("eval_opd_t5_small", 0, "opd_gsm8k_S-flan-t5-small_T-flan-t5-xl_expectation_tr_*"),
    ("eval_opd_t5_base",  1, "opd_gsm8k_S-flan-t5-base_T-flan-t5-xl_expectation_tr_*"),
    ("eval_opd_t5_large", 3, "opd_gsm8k_S-flan-t5-large_T-flan-t5-xl_expectation_tr_*"),
]


def find_latest(pattern: str) -> Path:
    """Return the most recently modified directory matching a glob pattern."""
    matches = sorted(SCRIPT_DIR.glob(pattern), key=lambda d: d.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No directory found matching: {SCRIPT_DIR / pattern}")
    return matches[-1]


def find_checkpoints(model_dir: Path) -> list[tuple[int | None, Path]]:
    """
    Return (step, path) for every checkpoint-N/ subdir, sorted by step.
    Appends (None, model_dir) only if the root directory contains a final model.
    """
    ckpts = sorted(
        [(int(d.name.split("-")[1]), d) for d in model_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda t: t[0],
    )
    if (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists():
        ckpts.append((None, model_dir))
    return ckpts


def build_eval_cmd(gpu: int, path: Path, step: int | None) -> str:
    tag = f"ckpt{step}" if step is not None else "final"
    log = path / f"eval_t5_size_ablation_{tag}.log"
    return (
        f"echo '=== Evaluating {path.name} ({tag}) ==='"
        f" && CUDA_VISIBLE_DEVICES={gpu} python -u {EVAL_T5}"
        f" --model_dir {path}"
        f" 2>&1 | tee {log}"
    )


def resolve_jobs() -> list[tuple[str, int, Path]]:
    return [
        (session, gpu, find_latest(pattern))
        for session, gpu, pattern in SPECS
    ]


def session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={name}"], capture_output=True
    ).returncode == 0


def launch(session: str, gpu: int, model_dir: Path) -> bool:
    if session_exists(session):
        print(
            f"[SKIP]  Session '{session}' already exists — "
            f"kill it first with: tmux kill-session -t {session}"
        )
        return False

    checkpoints = find_checkpoints(model_dir)
    if not checkpoints:
        print(f"[SKIP]  No checkpoints or final model found in {model_dir.name}")
        return False

    step_cmds = " && ".join(
        build_eval_cmd(gpu, path, step)
        for step, path in checkpoints
    )
    script_body = (
        f"#!/bin/bash\n"
        f"source $(conda info --base)/etc/profile.d/conda.sh\n"
        f"conda activate opd\n"
        f"cd {SCRIPT_DIR}\n"
        f"{step_cmds}\n"
        f"echo\n"
        f"echo '=== All checkpoints done. Session staying open. ==='\n"
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
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate T5 size ablation models on GSM8K using the flexible T5 protocol."
    )
    parser.add_argument(
        "--algo", choices=["gkd", "opd", "both"], default="both",
        help="Which algorithm to evaluate (default: both)",
    )
    args = parser.parse_args()

    if not EVAL_T5.exists():
        print(f"[ERROR] Eval script not found: {EVAL_T5}")
        sys.exit(1)

    try:
        jobs = resolve_jobs()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if args.algo != "both":
        jobs = [j for j in jobs if f"_{args.algo}_" in j[0]]

    print("Discovered model directories and their checkpoints:")
    for session, gpu, model_dir in jobs:
        ckpts = find_checkpoints(model_dir)
        if ckpts:
            ckpt_labels = ", ".join(
                f"ckpt{s}" if s is not None else "final" for s, _ in ckpts
            )
        else:
            ckpt_labels = "(none found)"
        print(f"  GPU {gpu}  {session:<24}  {model_dir.name}")
        print(f"            checkpoints: {ckpt_labels}")
    print()

    launched = 0
    for session, gpu, model_dir in jobs:
        if launch(session, gpu, model_dir):
            n = len(find_checkpoints(model_dir))
            print(f"[OK]    Session '{session}' started on GPU {gpu} ({n} evals) → {model_dir.name}")
            launched += 1

    print()
    print(f"Launched {launched}/{len(jobs)} sessions.")
    print()
    print("  GPU 0  →  eval_gkd_t5_small  eval_opd_t5_small  (concurrent)")
    print("  GPU 1  →  eval_gkd_t5_base   eval_opd_t5_base   (concurrent)")
    print("  GPU 2  →  eval_gkd_t5_large")
    print("  GPU 3  →  eval_opd_t5_large")
    print()
    print("Attach to a session with:  tmux attach -t <session>")
    print("List all sessions with:    tmux ls")
    print()
    print("Results saved as eval_gsm8k_t5.json inside each checkpoint directory.")


if __name__ == "__main__":
    main()
