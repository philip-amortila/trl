#!/usr/bin/env python3
"""
GKD (Generalized Knowledge Distillation) for seq2seq Flan-T5 on GSM8K.

Mirrors the logic of gkd_experiment.py but uses AutoModelForSeq2SeqLM and
Seq2SeqTrainer.  The GKD objective is the Generalized JSD between student and
teacher *decoder* logits on the student's on-policy trajectory.

Reference: Agarwal et al. 2024 – "On-Policy Distillation of Language Models:
Learning from Self-Generated Mistakes", Figure 1 (right): T5-Small/Base/Large
students, FT T5-XL teacher on GSM8K.

Usage (via env vars, same pattern as the Qwen scripts):
  STUDENT_MODEL=google/flan-t5-small  TEACHER_MODEL=google/flan-t5-xl  python -u gkd_t5_experiment.py
"""

import json
import os
import random
import re
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ─── Config ───────────────────────────────────────────────────────────────────
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "google/flan-t5-small")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "google/flan-t5-xl")

TRAIN_SAMPLES = int(os.environ.get("TRAIN_SAMPLES", "7473"))   # full GSM8K train
EVAL_SAMPLES  = int(os.environ.get("EVAL_SAMPLES",  "200"))

PER_DEVICE_BATCH_SIZE = int(os.environ.get("BATCH_SIZE",  "4"))
GRAD_ACCUM            = int(os.environ.get("GRAD_ACCUM",  "8"))
MAX_STEPS             = int(os.environ.get("MAX_STEPS",   "500"))
LR                    = float(os.environ.get("LR",        "1e-4"))
BETA                  = float(os.environ.get("BETA",      "0.5"))
LMBDA                 = float(os.environ.get("LMBDA",     "1.0"))
MAX_NEW_TOKENS        = int(os.environ.get("MAX_NEW_TOKENS",  "200"))
MAX_INPUT_LENGTH      = int(os.environ.get("MAX_INPUT_LENGTH", "512"))
MAX_TARGET_LENGTH     = int(os.environ.get("MAX_TARGET_LENGTH", "256"))

LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))
SAVE_STEPS    = int(os.environ.get("SAVE_STEPS",    "250"))


def _short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


_default_output_dir = (
    f"gkd_gsm8k"
    f"_S-{_short_name(STUDENT_MODEL)}"
    f"_T-{_short_name(TEACHER_MODEL)}"
    f"_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", _default_output_dir)

# ─── Prompt format ────────────────────────────────────────────────────────────

INPUT_PREFIX = (
    "Solve the math problem step by step. "
    "Write your final answer as #### <number>.\n\n"
)

def format_input(question: str) -> str:
    return INPUT_PREFIX + question.strip()


_ANS_RE = re.compile(r"####\s*([\-]?\d[\d,\.]*)")

def extract_answer(text: str) -> Optional[str]:
    m = _ANS_RE.search(text)
    return m.group(1).replace(",", "") if m else None


# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_datasets(tokenizer: AutoTokenizer, train_n: int, eval_n: int):
    ds = load_dataset("gsm8k", "main")
    train_raw = ds["train"].select(range(min(train_n, len(ds["train"]))))
    test_raw  = ds["test"].select( range(min(eval_n,  len(ds["test"]))))

    def tokenize(batch):
        model_inputs = tokenizer(
            [format_input(q) for q in batch["question"]],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,
        )
        # T5 uses a shared vocabulary; tokenize targets the same way
        targets = tokenizer(
            batch["answer"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = targets["input_ids"]
        return model_inputs

    train_tok = train_raw.map(tokenize, batched=True, remove_columns=train_raw.column_names)
    eval_tok  = test_raw.map( tokenize, batched=True, remove_columns=test_raw.column_names)
    return train_tok, eval_tok, ds["test"].select(range(min(eval_n, len(ds["test"]))))


# ─── GKD loss ─────────────────────────────────────────────────────────────────

def generalized_jsd_loss(
    student_logits: torch.Tensor,   # (B, T, V_s)
    teacher_logits: torch.Tensor,   # (B, T, V_t)
    labels: torch.Tensor,           # (B, T), -100 at padding
    beta: float = 0.5,
) -> torch.Tensor:
    """Generalized Jensen-Shannon Divergence loss (Eq. 1 of GKD paper)."""
    min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    s_log = F.log_softmax(student_logits[..., :min_vocab], dim=-1)
    t_log = F.log_softmax(teacher_logits[..., :min_vocab], dim=-1)

    beta_t = torch.tensor(beta, dtype=s_log.dtype, device=s_log.device)
    mix_log = torch.logsumexp(
        torch.stack([s_log + torch.log1p(-beta_t), t_log + torch.log(beta_t)]),
        dim=0,
    )
    kl_t = F.kl_div(mix_log, t_log, reduction="none", log_target=True)
    kl_s = F.kl_div(mix_log, s_log, reduction="none", log_target=True)
    jsd = (beta * kl_t + (1 - beta) * kl_s).sum(-1)   # (B, T)

    mask = labels != -100
    return jsd[mask].sum() / mask.sum().clamp(min=1)


# ─── Trainer ──────────────────────────────────────────────────────────────────

class GKDT5Trainer(Seq2SeqTrainer):
    """GKD trainer for encoder-decoder (T5-family) models.

    With probability ``lmbda`` each step, the student generates a decoder
    sequence on-policy; student and teacher logits on that trajectory are
    used to compute the Generalized JSD loss.
    """

    def __init__(
        self,
        teacher_model: AutoModelForSeq2SeqLM,
        lmbda: float = 1.0,
        beta: float = 0.5,
        max_new_tokens: int = 200,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        self.lmbda = lmbda
        self.beta = beta
        self.max_new_tokens = max_new_tokens

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if random.random() <= self.lmbda:
            with torch.no_grad():
                generated = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    top_k=0,
                    use_cache=True,
                )
            # T5.generate() prepends decoder_start_token_id (pad=0) at position 0
            on_policy_labels = generated[:, 1:].clone()
            _tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            pad_id = _tok.pad_token_id if (_tok is not None and _tok.pad_token_id is not None) else 0
            on_policy_labels[on_policy_labels == pad_id] = -100
            inputs = {**inputs, "labels": on_policy_labels}

        # Forward: T5 creates decoder_input_ids from labels internally
        student_out = model(**inputs)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_out = self.teacher_model(**inputs)

        loss = generalized_jsd_loss(
            student_out.logits,
            teacher_out.logits.to(student_out.logits.device),
            inputs["labels"],
            beta=self.beta,
        )
        return (loss, student_out) if return_outputs else loss


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def gsm8k_exact_match(
    model,
    tokenizer,
    raw_ds,
    max_new_tokens: int = 200,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    n = min(limit, len(raw_ds)) if limit else len(raw_ds)
    correct = parsed = 0
    for i in range(n):
        enc = tokenizer(
            format_input(raw_ds[i]["question"]),
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_answer(text)
        gold = extract_answer(raw_ds[i]["answer"])
        if pred is not None:
            parsed += 1
        if pred is not None and pred == gold:
            correct += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}]  acc={correct/(i+1):.3f}  parse={parsed/(i+1):.3f}")
    return {"n": n, "exact_match": correct / n, "parse_rate": parsed / n, "correct": correct}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"STUDENT_MODEL={STUDENT_MODEL}")
    print(f"TEACHER_MODEL={TEACHER_MODEL}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    print(f"LR={LR}  BETA={BETA}  LMBDA={LMBDA}  MAX_STEPS={MAX_STEPS}")
    print(f"BATCH={PER_DEVICE_BATCH_SIZE}  GRAD_ACCUM={GRAD_ACCUM}  MAX_NEW_TOKENS={MAX_NEW_TOKENS}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump({
            "method": "gkd",
            "student": STUDENT_MODEL,
            "teacher": TEACHER_MODEL,
            "lr": LR, "beta": BETA, "lmbda": LMBDA,
            "max_steps": MAX_STEPS,
            "batch_size": PER_DEVICE_BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "max_new_tokens": MAX_NEW_TOKENS,
        }, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    student = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_MODEL, torch_dtype=torch_dtype)
    teacher = AutoModelForSeq2SeqLM.from_pretrained(TEACHER_MODEL, torch_dtype=torch_dtype)

    train_ds, eval_ds, raw_test = build_datasets(tokenizer, TRAIN_SAMPLES, EVAL_SAMPLES)

    # model=None: do not add decoder_input_ids; T5 creates them from labels internally
    collator = DataCollatorForSeq2Seq(tokenizer, model=None, label_pad_token_id=-100, pad_to_multiple_of=8)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        max_steps=MAX_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        eval_strategy="no",
        do_train=True,
        report_to=["none"],
        bf16=torch.cuda.is_available(),
        predict_with_generate=False,
        remove_unused_columns=False,
    )

    trainer = GKDT5Trainer(
        teacher_model=teacher,
        lmbda=LMBDA,
        beta=BETA,
        max_new_tokens=MAX_NEW_TOKENS,
        model=student,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    pre = gsm8k_exact_match(student, tokenizer, raw_test, max_new_tokens=MAX_NEW_TOKENS, limit=50)
    print("Pre-train exact match:", pre)

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    post = gsm8k_exact_match(student, tokenizer, raw_test, max_new_tokens=MAX_NEW_TOKENS)
    print("Post-train exact match:", post)

    with open(os.path.join(OUTPUT_DIR, "eval_pre.json"),  "w") as f: json.dump(pre,  f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "eval_post.json"), "w") as f: json.dump(post, f, indent=2)

    history = trainer.state.log_history
    with open(os.path.join(OUTPUT_DIR, "log_history.jsonl"), "w") as f:
        for e in history:
            f.write(json.dumps(e) + "\n")

    print(f"Done. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
