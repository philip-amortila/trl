#!/usr/bin/env python3
"""
Evaluate all checkpoints of the 12 models from the student-size ablation
(3 sizes × 2 algorithms × 2 datasets = 12 runs).

For each model the script evaluates every checkpoint-N/ subdirectory found
at launch time, then the final root model, all sequentially inside one tmux
session.  This means the session stays alive until all checkpoints are done.

GSM8K-trained models  → eval_gsm8k.py        (strict + flex parsing)
MATH500-trained models → eval_math_benchmarks.py --benchmarks math500
                         (includes per-category breakdown)

Teacher: Qwen/Qwen2-7B-Instruct
Students: Qwen2-0.5B-Instruct, Qwen2-1.5B-Instruct, Qwen2.5-3B-Instruct
Algorithms: GKD, OPD (expectation + trust-region)

GPU layout (one session per model, checkpoints run sequentially):
  GPU 0  →  gkd_gsm8k_0.5B   opd_gsm8k_0.5B
             gkd_math500_3B    opd_math500_3B
  GPU 4  →  gkd_gsm8k_1.5B   opd_gsm8k_1.5B
             gkd_math500_1.5B  opd_math500_1.5B
  GPU 6  →  gkd_gsm8k_3B     opd_gsm8k_3B
             gkd_math500_0.5B  opd_math500_0.5B
"""

import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
EVAL_GSM8K = SCRIPT_DIR / "eval_gsm8k.py"
EVAL_MATH  = SCRIPT_DIR / "eval_math_benchmarks.py"
TEACHER    = "T-Qwen2-7B-Instruct"


def find_latest(pattern: str) -> Path:
    """Return the most recently modified directory matching a glob pattern."""
    matches = sorted(SCRIPT_DIR.glob(pattern), key=lambda d: d.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No directory found matching: {SCRIPT_DIR / pattern}")
    return matches[-1]


def find_checkpoints(model_dir: Path) -> list[tuple[int | None, Path]]:
    """
    Return (step, path) for every checkpoint-N/ subdir found, sorted by step,
    followed by (None, model_dir) for the final root model.
    """
    ckpts = sorted(
        [(int(d.name.split("-")[1]), d) for d in model_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda t: t[0],
    )
    ckpts.append((None, model_dir))
    return ckpts


def build_eval_cmd(gpu: int, path: Path, eval_script: Path, extra_args: str, step: int | None) -> str:
    """Return a single shell command that evaluates `path` and tees the log."""
    tag = f"ckpt{step}" if step is not None else "final"
    log = path / f"eval_size_ablation_{tag}.log"
    return (
        f"echo '=== Evaluating {path.name} ({tag}) ==='"
        f" && CUDA_VISIBLE_DEVICES={gpu} python -u {eval_script}"
        f" --model_dir {path} {extra_args}"
        f" 2>&1 | tee {log}"
    )


def resolve_jobs() -> list[tuple[str, int, Path, Path, str]]:
    """
    Return (session_name, gpu, model_dir, eval_script, extra_args) for all 12 models.
    """
    jobs = []

    gsm8k_specs = [
        ("eval_gkd_gsm8k_05B",  0, f"gkd_gsm8k_S-Qwen2-0.5B-Instruct_{TEACHER}_*"),
        ("eval_opd_gsm8k_05B",  0, f"opd_gsm8k_S-Qwen2-0.5B-Instruct_{TEACHER}_expectation_tr_L*_buf*"),
        ("eval_gkd_gsm8k_15B",  4, f"gkd_gsm8k_S-Qwen2-1.5B-Instruct_{TEACHER}_*"),
        ("eval_opd_gsm8k_15B",  4, f"opd_gsm8k_S-Qwen2-1.5B-Instruct_{TEACHER}_expectation_tr_L*_buf*"),
        ("eval_gkd_gsm8k_3B",   6, f"gkd_gsm8k_S-Qwen2.5-3B-Instruct_{TEACHER}_*"),
        ("eval_opd_gsm8k_3B",   6, f"opd_gsm8k_S-Qwen2.5-3B-Instruct_{TEACHER}_expectation_tr_L*_buf*"),
    ]
    for session, gpu, pattern in gsm8k_specs:
        model_dir = find_latest(pattern)
        jobs.append((session, gpu, model_dir, EVAL_GSM8K, ""))

    math500_specs = [
        ("eval_gkd_math500_05B", 6, f"gkd_math500_S-Qwen2-0.5B-Instruct_{TEACHER}_*"),
        ("eval_opd_math500_05B", 6, f"opd_math500_S-Qwen2-0.5B-Instruct_{TEACHER}_expectation_tr_L*_buf*"),
        ("eval_gkd_math500_15B", 4, f"gkd_math500_S-Qwen2-1.5B-Instruct_{TEACHER}_*"),
        ("eval_opd_math500_15B", 4, f"opd_math500_S-Qwen2-1.5B-Instruct_{TEACHER}_expectation_tr_L*_buf*"),
        ("eval_gkd_math500_3B",  0, f"gkd_math500_S-Qwen2.5-3B-Instruct_{TEACHER}_*"),
        ("eval_opd_math500_3B",  0, f"opd_math500_S-Qwen2.5-3B-Instruct_{TEACHER}_expectation_tr_L*_buf*"),
    ]
    for session, gpu, pattern in math500_specs:
        model_dir = find_latest(pattern)
        jobs.append((session, gpu, model_dir, EVAL_MATH, "--benchmarks math500"))

    return jobs


def session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={name}"], capture_output=True
    ).returncode == 0


def launch(session: str, gpu: int, model_dir: Path, eval_script: Path, extra_args: str) -> bool:
    """Create one tmux session that evaluates every checkpoint sequentially."""
    if session_exists(session):
        print(
            f"[SKIP]  Session '{session}' already exists — "
            f"kill it first with: tmux kill-session -t {session}"
        )
        return False

    checkpoints = find_checkpoints(model_dir)
    step_cmds = " && ".join(
        build_eval_cmd(gpu, path, eval_script, extra_args, step)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", choices=["gkd", "opd", "both"], default="both",
        help="Which algorithm to evaluate (default: both)",
    )
    args = parser.parse_args()

    for script in (EVAL_GSM8K, EVAL_MATH):
        if not script.exists():
            print(f"[ERROR] Eval script not found: {script}")
            sys.exit(1)

    try:
        jobs = resolve_jobs()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if args.algo != "both":
        jobs = [j for j in jobs if j[0].startswith(f"eval_{args.algo}_")]

    print("Discovered model directories and their checkpoints:")
    for session, gpu, model_dir, eval_script, _ in jobs:
        ckpts = find_checkpoints(model_dir)
        ckpt_labels = ", ".join(
            f"ckpt{s}" if s is not None else "final" for s, _ in ckpts
        )
        print(f"  GPU {gpu}  {session:<26}  {model_dir.name}")
        print(f"           checkpoints: {ckpt_labels}")
    print()

    launched = 0
    for session, gpu, model_dir, eval_script, extra_args in jobs:
        if launch(session, gpu, model_dir, eval_script, extra_args):
            n = len(find_checkpoints(model_dir))
            print(f"[OK]    Session '{session}' started on GPU {gpu} ({n} checkpoints) → {model_dir.name}")
            launched += 1

    print()
    print(f"Launched {launched}/{len(jobs)} sessions.")
    print("Attach to a session with:  tmux attach -t <session>")
    print("List all sessions with:    tmux ls")


if __name__ == "__main__":
    main()
