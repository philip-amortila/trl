#!/usr/bin/env python3
"""
OPD (On-Policy Distillation) for seq2seq Flan-T5 on GSM8K.

Implements the OPD expectation + trust-region variant (Algorithm 4 with
PPO clipped surrogate) for encoder-decoder T5 models.  This mirrors
opd_experiment.py but uses AutoModelForSeq2SeqLM and Seq2SeqTrainer.

Reference: Agarwal et al. 2024 – "On-Policy Distillation of Language Models:
Learning from Self-Generated Mistakes", Figure 1 (right): T5-Small/Base/Large
students, FT T5-XL teacher on GSM8K.

Usage (via env vars):
  STUDENT_MODEL=google/flan-t5-small  TEACHER_MODEL=google/flan-t5-xl  python -u opd_t5_experiment.py
"""

import json
import os
import random as _random
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

TRAIN_SAMPLES = int(os.environ.get("TRAIN_SAMPLES", "7473"))
EVAL_SAMPLES  = int(os.environ.get("EVAL_SAMPLES",  "200"))

PER_DEVICE_BATCH_SIZE = int(os.environ.get("BATCH_SIZE",  "4"))
GRAD_ACCUM            = int(os.environ.get("GRAD_ACCUM",  "8"))
MAX_STEPS             = int(os.environ.get("MAX_STEPS",   "500"))
LR                    = float(os.environ.get("LR",        "1e-5"))
LMBDA                 = float(os.environ.get("LMBDA",     "1.0"))
MAX_NEW_TOKENS        = int(os.environ.get("MAX_NEW_TOKENS",  "200"))
MAX_INPUT_LENGTH      = int(os.environ.get("MAX_INPUT_LENGTH", "512"))
MAX_TARGET_LENGTH     = int(os.environ.get("MAX_TARGET_LENGTH", "256"))

OPD_MODE            = os.environ.get("OPD_MODE",          "expectation")
NUM_INNER_STEPS     = int(os.environ.get("NUM_INNER_STEPS",  "10"))
REPLAY_BUFFER_SIZE  = int(os.environ.get("REPLAY_BUFFER_SIZE", "10"))
TRUST_REGION        = os.environ.get("TRUST_REGION", "1") == "1"
PPO_CLIP_EPS        = float(os.environ.get("PPO_CLIP_EPS", "0.2"))

LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))
SAVE_STEPS    = int(os.environ.get("SAVE_STEPS",    "250"))


def _short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


_default_output_dir = (
    f"opd_gsm8k"
    f"_S-{_short_name(STUDENT_MODEL)}"
    f"_T-{_short_name(TEACHER_MODEL)}"
    f"_{OPD_MODE}"
    f"{'_tr' if TRUST_REGION else ''}"
    f"_L{NUM_INNER_STEPS}"
    f"_buf{REPLAY_BUFFER_SIZE}"
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


# ─── OPD loss ─────────────────────────────────────────────────────────────────

def opd_loss(
    student_logits: torch.Tensor,   # (B, T, V_s)
    teacher_logits: torch.Tensor,   # (B, T, V_t)
    labels: torch.Tensor,           # (B, T), -100 at padding — student on-policy tokens
    mode: str = "expectation",
    trust_region: bool = False,
    ppo_clip_eps: float = 0.2,
    old_log_probs: Optional[torch.Tensor] = None,  # (B, T) behaviour-policy log-probs
) -> torch.Tensor:
    """
    OPD loss for seq2seq decoder logits.

    The math is identical to OPDTrainer.opd_loss; the difference is that
    here student_logits / teacher_logits are already the full decoder
    output, so no prompt-length slicing is needed.
    """
    min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    s_log = F.log_softmax(student_logits[..., :min_vocab], dim=-1)  # (B, T, V)
    t_log = F.log_softmax(teacher_logits[..., :min_vocab], dim=-1)  # (B, T, V)

    if mode == "expectation":
        B, T, V = t_log.shape
        expert_actions = torch.multinomial(
            t_log.exp().view(B * T, V), num_samples=1
        ).view(B, T)

        log_pi_e_expert = t_log.gather(-1, expert_actions.unsqueeze(-1)).squeeze(-1)

        if trust_region and old_log_probs is not None:
            student_actions = labels.clone()
            student_actions[student_actions == -100] = 0
            log_pi_e_student = t_log.gather(-1, student_actions.unsqueeze(-1)).squeeze(-1)
            log_pi_s_student = s_log.gather(-1, student_actions.unsqueeze(-1)).squeeze(-1)
            advantage = (log_pi_e_expert - log_pi_e_student).detach()
            ratio = (log_pi_s_student - old_log_probs).exp()
            clipped = ratio.clamp(1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps)
            loss = torch.max(ratio * advantage, clipped * advantage)
        else:
            expected_log_pi_e = (s_log.exp() * t_log).sum(-1)
            loss = log_pi_e_expert - expected_log_pi_e

    elif mode == "stochastic":
        B, T, V = t_log.shape
        expert_actions = torch.multinomial(
            t_log.exp().view(B * T, V), num_samples=1
        ).view(B, T)
        log_pi_e_expert = t_log.gather(-1, expert_actions.unsqueeze(-1)).squeeze(-1)

        student_actions = labels.clone()
        student_actions[student_actions == -100] = 0
        log_pi_e_student = t_log.gather(-1, student_actions.unsqueeze(-1)).squeeze(-1)
        log_pi_s_student = s_log.gather(-1, student_actions.unsqueeze(-1)).squeeze(-1)
        advantage = (log_pi_e_expert - log_pi_e_student).detach()

        if trust_region and old_log_probs is not None:
            ratio = (log_pi_s_student - old_log_probs).exp()
            clipped = ratio.clamp(1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps)
            loss = torch.max(ratio * advantage, clipped * advantage)
        else:
            loss = advantage * log_pi_s_student

    elif mode == "entropy_baseline":
        neg_teacher_entropy = (t_log.exp() * t_log).sum(-1)
        student_actions = labels.clone()
        student_actions[student_actions == -100] = 0
        log_pi_e_student = t_log.gather(-1, student_actions.unsqueeze(-1)).squeeze(-1)
        log_pi_s_student = s_log.gather(-1, student_actions.unsqueeze(-1)).squeeze(-1)
        advantage = (neg_teacher_entropy - log_pi_e_student).detach()

        if trust_region and old_log_probs is not None:
            ratio = (log_pi_s_student - old_log_probs).exp()
            clipped = ratio.clamp(1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps)
            loss = torch.max(ratio * advantage, clipped * advantage)
        else:
            loss = advantage * log_pi_s_student

    else:
        raise ValueError(f"Unknown OPD mode: {mode!r}")

    mask = labels != -100
    loss = loss[mask]
    return loss.sum() / mask.sum().clamp(min=1)


# ─── Trainer ──────────────────────────────────────────────────────────────────

class OPDT5Trainer(Seq2SeqTrainer):
    """OPD trainer for encoder-decoder (T5-family) models.

    Implements the outer loop of Algorithm 4: on-policy generation,
    replay buffer, L inner gradient steps, and optional PPO trust region.
    """

    def __init__(
        self,
        teacher_model: AutoModelForSeq2SeqLM,
        lmbda: float = 1.0,
        max_new_tokens: int = 200,
        mode: str = "expectation",
        num_inner_steps: int = 10,
        replay_buffer_size: int = 10,
        trust_region: bool = True,
        ppo_clip_eps: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        self.lmbda = lmbda
        self.max_new_tokens = max_new_tokens
        self.mode = mode
        self.num_inner_steps = num_inner_steps
        self.trust_region = trust_region
        self.ppo_clip_eps = ppo_clip_eps
        self._replay_buffer: deque = deque(maxlen=max(replay_buffer_size, 1))

    def _collect_on_policy(self, model, inputs: dict) -> dict:
        """Generate student trajectory and optionally record behaviour log-probs."""
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_k=0,
                use_cache=True,
            )
        # Strip the leading decoder_start_token (T5 prepends pad=0)
        on_policy_labels = generated[:, 1:].clone()
        _tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        pad_id = _tok.pad_token_id if (_tok is not None and _tok.pad_token_id is not None) else 0
        on_policy_labels[on_policy_labels == pad_id] = -100

        new_inputs = {**inputs, "labels": on_policy_labels}

        if self.trust_region:
            with torch.no_grad():
                old_out = model(**new_inputs)
            old_lp_all = F.log_softmax(old_out.logits, dim=-1)   # (B, T, V)
            sa = on_policy_labels.clone()
            sa[sa == -100] = 0
            old_lp = old_lp_all.gather(-1, sa.unsqueeze(-1)).squeeze(-1)   # (B, T)
            old_lp[on_policy_labels == -100] = 0.0
            new_inputs["old_log_probs"] = old_lp

        return new_inputs

    def _push_to_replay(self, inputs: dict) -> None:
        self._replay_buffer.append(
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
             for k, v in inputs.items()}
        )

    def _sample_from_replay(self) -> dict:
        return self._prepare_inputs(dict(_random.choice(self._replay_buffer)))

    def training_step(self, model, inputs, num_items_in_batch=None):
        # 1. Generate on-policy trajectory
        if _random.random() <= self.lmbda:
            inputs = self._collect_on_policy(model, inputs)

        # 2. Push to replay buffer
        self._push_to_replay(inputs)

        # 3. L inner gradient steps over replay buffer
        model.train()
        total_loss = torch.tensor(0.0, device=self.args.device)
        total_grad_norm = 0.0

        for _ in range(self.num_inner_steps):
            batch = self._sample_from_replay()

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, batch, num_items_in_batch=num_items_in_batch)

            self.accelerator.backward(loss)
            if self.args.max_grad_norm and self.args.max_grad_norm > 0:
                grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                total_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.detach()

        self._grad_norm = total_grad_norm / self.num_inner_steps
        return total_loss / self.num_inner_steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_out = model(**{k: v for k, v in inputs.items() if k != "old_log_probs"})
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_out = self.teacher_model(**{k: v for k, v in inputs.items() if k != "old_log_probs"})

        loss = opd_loss(
            student_logits=student_out.logits,
            teacher_logits=teacher_out.logits.to(student_out.logits.device),
            labels=inputs["labels"],
            mode=self.mode,
            trust_region=self.trust_region,
            ppo_clip_eps=self.ppo_clip_eps,
            old_log_probs=inputs.get("old_log_probs"),
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
    print(f"OPD_MODE={OPD_MODE}  TRUST_REGION={TRUST_REGION}  PPO_CLIP_EPS={PPO_CLIP_EPS}")
    print(f"NUM_INNER_STEPS={NUM_INNER_STEPS}  REPLAY_BUFFER_SIZE={REPLAY_BUFFER_SIZE}")
    print(f"LR={LR}  LMBDA={LMBDA}  MAX_STEPS={MAX_STEPS}")
    print(f"BATCH={PER_DEVICE_BATCH_SIZE}  GRAD_ACCUM={GRAD_ACCUM}  MAX_NEW_TOKENS={MAX_NEW_TOKENS}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump({
            "method": "opd",
            "student": STUDENT_MODEL,
            "teacher": TEACHER_MODEL,
            "opd_mode": OPD_MODE,
            "trust_region": TRUST_REGION,
            "ppo_clip_eps": PPO_CLIP_EPS,
            "num_inner_steps": NUM_INNER_STEPS,
            "replay_buffer_size": REPLAY_BUFFER_SIZE,
            "lr": LR, "lmbda": LMBDA,
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

    trainer = OPDT5Trainer(
        teacher_model=teacher,
        lmbda=LMBDA,
        max_new_tokens=MAX_NEW_TOKENS,
        mode=OPD_MODE,
        num_inner_steps=NUM_INNER_STEPS,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        trust_region=TRUST_REGION,
        ppo_clip_eps=PPO_CLIP_EPS,
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
