#!/usr/bin/env python3
"""
Train GKD and OPD-expectation sequentially in one process, sharing a single
teacher loaded on GPUs 0+1.

GPU layout
  GPU 0+1  →  teacher  (57B MoE, ~114 GB bfloat16)
  GPU 2    →  student  (used by both GKD and OPD in sequence)

The teacher is loaded once and reused for both training runs.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.gkd import GKDConfig, GKDTrainer
from trl.experimental.gkd.opd_trainer import OPDTrainer


# ── Config ────────────────────────────────────────────────────────────────────

TEACHER_MODEL  = os.environ.get("TEACHER_MODEL",  "Qwen/Qwen2-57B-A14B-Instruct")
STUDENT_MODEL  = os.environ.get("STUDENT_MODEL",  "Qwen/Qwen2-0.5B-Instruct")
DATASET_NAME   = os.environ.get("DATASET_NAME",   "gsm8k")
DATASET_CONFIG = os.environ.get("DATASET_CONFIG", "main")

TRAIN_SAMPLES = int(os.environ.get("TRAIN_SAMPLES", "2000"))
EVAL_SAMPLES  = int(os.environ.get("EVAL_SAMPLES",  "256"))
MAX_STEPS     = int(os.environ.get("MAX_STEPS",     "800"))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE",    "1"))
GRAD_ACCUM    = int(os.environ.get("GRAD_ACCUM",    "8"))
LR            = float(os.environ.get("LR",          "2e-6"))

NUM_INNER_STEPS    = int(os.environ.get("NUM_INNER_STEPS",    "10"))
REPLAY_BUFFER_SIZE = int(os.environ.get("REPLAY_BUFFER_SIZE", "10"))

LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))
EVAL_STEPS    = int(os.environ.get("EVAL_STEPS",    "100"))

# GPU indices
TEACHER_GPUS  = [0, 1]   # teacher spread across these two
STUDENT_GPU   = 2         # single student GPU, reused for GKD then OPD

TEACHER_MEM_PER_GPU = "75GiB"   # leave headroom on each A100-80GB

TS = datetime.now().strftime("%Y%m%d_%H%M%S")

def _short(model_id: str) -> str:
    return model_id.split("/")[-1]

GKD_OUTPUT_DIR = os.environ.get(
    "GKD_OUTPUT_DIR",
    f"gkd_gsm8k_S-{_short(STUDENT_MODEL)}_T-{_short(TEACHER_MODEL)}_{TS}",
)
OPD_OUTPUT_DIR = os.environ.get(
    "OPD_OUTPUT_DIR",
    f"opd_gsm8k_S-{_short(STUDENT_MODEL)}_T-{_short(TEACHER_MODEL)}_expectation_tr_L{NUM_INNER_STEPS}_buf{REPLAY_BUFFER_SIZE}_{TS}",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem and give the final answer.\n"
    "Put the final numeric answer on a line by itself in the format: #### <number>\n"
)

_ANS_RE = re.compile(r"####\s*([\-]?\d[\d,\.]*)")


def _parse_answer(text: str) -> Optional[str]:
    m = _ANS_RE.search(text)
    return m.group(1).strip().replace(",", "") if m else None


def _to_messages(question: str, answer: Optional[str]) -> List[Dict[str, str]]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer.strip()})
    return msgs


def build_datasets(train_n: int, eval_n: int) -> Tuple[Dataset, Dataset, Dataset]:
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
    train_raw = ds["train"].select(range(min(train_n, len(ds["train"]))))
    eval_raw  = ds["test"].select(range(min(eval_n,  len(ds["test"]))))

    train_msgs = [_to_messages(ex["question"], ex["answer"]) for ex in train_raw]
    eval_msgs  = [_to_messages(ex["question"], ex["answer"]) for ex in eval_raw]

    return (
        Dataset.from_dict({"messages": train_msgs}),
        Dataset.from_dict({"messages": eval_msgs}),
        eval_raw,
    )


@torch.no_grad()
def gsm8k_exact_match(model, tokenizer, eval_raw, limit=256) -> Dict[str, Any]:
    model.eval()
    n = min(limit, len(eval_raw))
    correct = parsed = 0
    for i in range(n):
        gold = _parse_answer(eval_raw[i]["answer"])
        prompt = tokenizer.apply_chat_template(
            _to_messages(eval_raw[i]["question"], None),
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        pred = _parse_answer(tokenizer.decode(out[0], skip_special_tokens=True))
        if pred is not None:
            parsed += 1
        if pred is not None and gold is not None and pred == gold:
            correct += 1
    return {"n": n, "exact_match": correct / n, "parse_rate": parsed / n}


def save_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "log_history.jsonl"), "w") as f:
        for e in history:
            f.write(json.dumps(e) + "\n")


def make_base_args(output_dir: str) -> GKDConfig:
    args = GKDConfig(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        max_steps=MAX_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="no",
        do_train=True,
        do_eval=True,
        report_to=["wandb"],
        bf16=True,
        lmbda=1.0,
    )
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

def train_gkd(teacher, tokenizer, train_dataset, eval_dataset, eval_raw) -> dict:
    print("=" * 60)
    print(f"Starting GKD  →  {GKD_OUTPUT_DIR}")
    print("=" * 60)

    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=torch.bfloat16, device_map={"": STUDENT_GPU},
    )
    os.makedirs(GKD_OUTPUT_DIR, exist_ok=True)
    json.dump({"method": "gkd", "student": STUDENT_MODEL, "teacher": TEACHER_MODEL,
               "max_steps": MAX_STEPS, "lr": LR},
              open(os.path.join(GKD_OUTPUT_DIR, "run_config.json"), "w"), indent=2)

    trainer = GKDTrainer(
        model=student,
        teacher_model=teacher,
        args=make_base_args(GKD_OUTPUT_DIR),
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(GKD_OUTPUT_DIR)
    save_history(trainer.state.log_history, GKD_OUTPUT_DIR)

    ev = gsm8k_exact_match(student, tokenizer, eval_raw)
    json.dump(ev, open(os.path.join(GKD_OUTPUT_DIR, "eval_post.json"), "w"), indent=2)
    print(f"GKD done  exact_match={ev['exact_match']:.4f}\n")

    del student, trainer
    torch.cuda.empty_cache()
    return ev


def train_opd(teacher, tokenizer, train_dataset, eval_dataset, eval_raw) -> dict:
    print("=" * 60)
    print(f"Starting OPD-expectation (TR)  →  {OPD_OUTPUT_DIR}")
    print("=" * 60)

    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=torch.bfloat16, device_map={"": STUDENT_GPU},
    )
    os.makedirs(OPD_OUTPUT_DIR, exist_ok=True)
    json.dump({"method": "opd_expectation_tr", "student": STUDENT_MODEL, "teacher": TEACHER_MODEL,
               "num_inner_steps": NUM_INNER_STEPS, "replay_buffer_size": REPLAY_BUFFER_SIZE,
               "max_steps": MAX_STEPS, "lr": LR},
              open(os.path.join(OPD_OUTPUT_DIR, "run_config.json"), "w"), indent=2)

    trainer = OPDTrainer(
        model=student,
        teacher_model=teacher,
        args=make_base_args(OPD_OUTPUT_DIR),
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        mode="expectation",
        num_inner_steps=NUM_INNER_STEPS,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        trust_region=True,
        use_correction=False,
    )
    trainer.train()
    trainer.save_model(OPD_OUTPUT_DIR)
    save_history(trainer.state.log_history, OPD_OUTPUT_DIR)

    ev = gsm8k_exact_match(student, tokenizer, eval_raw)
    json.dump(ev, open(os.path.join(OPD_OUTPUT_DIR, "eval_post.json"), "w"), indent=2)
    print(f"OPD done  exact_match={ev['exact_match']:.4f}\n")

    del student, trainer
    torch.cuda.empty_cache()
    return ev


def main():
    print(f"Teacher     : {TEACHER_MODEL}  (GPUs {TEACHER_GPUS})")
    print(f"Student     : {STUDENT_MODEL}  (GPU {STUDENT_GPU})")
    print(f"GKD out     : {GKD_OUTPUT_DIR}")
    print(f"OPD out     : {OPD_OUTPUT_DIR}")

    train_dataset, eval_dataset, eval_raw = build_datasets(TRAIN_SAMPLES, EVAL_SAMPLES)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load teacher once on GPUs 0+1 ─────────────────────────────────────────
    # Block all GPUs not in TEACHER_GPUS so accelerate can't spill onto them.
    n_visible = torch.cuda.device_count()
    teacher_max_mem = {i: ("0GiB" if i not in TEACHER_GPUS else TEACHER_MEM_PER_GPU)
                       for i in range(n_visible)}

    print(f"\nLoading teacher {TEACHER_MODEL} ...")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=teacher_max_mem,
    )
    teacher.eval()
    print("Teacher loaded.\n")

    # ── Train GKD then OPD sequentially, teacher stays loaded ─────────────────
    gkd_ev = train_gkd(teacher, tokenizer, train_dataset, eval_dataset, eval_raw)
    opd_ev = train_opd(teacher, tokenizer, train_dataset, eval_dataset, eval_raw)

    print("All done.")
    print(f"  GKD  flex={gkd_ev['exact_match']:.4f}  →  {GKD_OUTPUT_DIR}")
    print(f"  OPD  flex={opd_ev['exact_match']:.4f}  →  {OPD_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
