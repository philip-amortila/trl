"""
Single-student GKD training job.

Each invocation trains one student against one teacher independently —
intended to be run as a self-contained SLURM job on its own GPU.

Key env vars
  STUDENT_MODEL   HF model ID for the student   (required)
  TEACHER_MODEL   HF model ID for the teacher   (default: Qwen/Qwen3.5-9B)
  OUTPUT_DIR      Root directory for outputs     (default: gkd_qwen35_experiments)
  MAX_STEPS       Training steps                 (default: 600)
  SAVE_STEPS      Checkpoint interval            (default: 100)
  BATCH_SIZE      Per-device train batch size    (default: 1)
  GRAD_ACCUM      Gradient accumulation steps    (default: 8)
  LR              Learning rate                  (default: 2e-6)
  BETA            GKD beta                       (default: 0.5)
  LOGGING_STEPS                                  (default: 10)
"""

import os
import sys
import json

import torch

# gkd_experiment.py overwrites HUGGING_FACE_HUB_TOKEN with a placeholder at
# module level — preserve the real value before importing.
_real_hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gkd_experiment import (  # noqa: E402
    build_gsm8k_datasets,
    gsm8k_exact_match,
    plot_trainer_history,
    DATASET_NAME,
    DATASET_CONFIG,
    TRAIN_SAMPLES,
    EVAL_SAMPLES,
    MAX_NEW_TOKENS_EVAL,
)

if _real_hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _real_hf_token

from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl.experimental.gkd import GKDConfig, GKDTrainer  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------
STUDENT_MODEL = os.environ.get("STUDENT_MODEL")
if not STUDENT_MODEL:
    raise ValueError("STUDENT_MODEL env var is required (e.g. Qwen/Qwen3.5-4B)")

TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen3.5-9B")

MAX_STEPS       = int(os.environ.get("MAX_STEPS",      "600"))
SAVE_STEPS      = int(os.environ.get("SAVE_STEPS",     "100"))
LOGGING_STEPS   = int(os.environ.get("LOGGING_STEPS",  "10"))
LR              = float(os.environ.get("LR",           "2e-6"))
BETA            = float(os.environ.get("BETA",         "0.5"))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE",     "1"))
GRAD_ACCUM      = int(os.environ.get("GRAD_ACCUM",     "8"))

_short_student = STUDENT_MODEL.split("/")[-1]
_short_teacher = TEACHER_MODEL.split("/")[-1]
_base           = os.environ.get("OUTPUT_DIR", "gkd_qwen35_experiments")
OUTPUT_DIR      = os.path.join(_base, f"S-{_short_student}_T-{_short_teacher}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device        : {device}  dtype={torch_dtype}")
    print(f"Student       : {STUDENT_MODEL}")
    print(f"Teacher       : {TEACHER_MODEL}")
    print(f"Steps         : {MAX_STEPS}  (checkpoint every {SAVE_STEPS})")
    print(f"Output        : {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Persist run configuration for reproducibility
    run_config = {
        "student_model": STUDENT_MODEL,
        "teacher_model": TEACHER_MODEL,
        "max_steps": MAX_STEPS,
        "save_steps": SAVE_STEPS,
        "lr": LR,
        "beta": BETA,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
    }
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Datasets
    print("Loading datasets …")
    train_dataset, eval_dataset = build_gsm8k_datasets(TRAIN_SAMPLES, EVAL_SAMPLES)
    raw     = load_dataset(DATASET_NAME, DATASET_CONFIG)
    eval_raw = raw["test"].select(range(min(EVAL_SAMPLES, len(raw["test"]))))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Student model
    print(f"Loading student: {STUDENT_MODEL} …")
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Teacher model (loaded independently — no sharing across jobs)
    print(f"Loading teacher: {TEACHER_MODEL} …")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    teacher_model.eval()

    # Pre-training accuracy snapshot
    pre = gsm8k_exact_match(
        model, tokenizer, eval_raw, max_new_tokens=MAX_NEW_TOKENS_EVAL, limit=64
    )
    print(f"Pre-train  exact_match={pre['exact_match']:.3f}  parse_rate={pre['parse_rate']:.3f}")

    # Training
    args = GKDConfig(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=LOGGING_STEPS,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=None,  # keep all checkpoints
        do_train=True,
        do_eval=False,
        report_to=["none"],
        bf16=torch.cuda.is_available(),
        lmbda=1.0,
        beta=BETA,
        push_to_hub=False,
    )

    trainer = GKDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    history = trainer.state.log_history
    with open(os.path.join(OUTPUT_DIR, "log_history.jsonl"), "w") as f:
        for entry in history:
            f.write(json.dumps(entry) + "\n")
    plot_trainer_history(history, OUTPUT_DIR)

    # Post-training accuracy snapshot
    post = gsm8k_exact_match(
        model, tokenizer, eval_raw, max_new_tokens=MAX_NEW_TOKENS_EVAL, limit=64
    )
    print(f"Post-train exact_match={post['exact_match']:.3f}  parse_rate={post['parse_rate']:.3f}")

    with open(os.path.join(OUTPUT_DIR, "eval_pre.json"),  "w") as f:
        json.dump(pre,  f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "eval_post.json"), "w") as f:
        json.dump(post, f, indent=2)

    delta = post["exact_match"] - pre["exact_match"]
    sign  = "+" if delta >= 0 else ""
    print(f"Done.  Δ exact_match = {sign}{delta:.3f}  |  outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
