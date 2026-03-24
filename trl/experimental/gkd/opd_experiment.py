# gkd_experiment.py
import os
import re
import json
#from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.gkd import GKDConfig
from trl.experimental.gkd.opd_trainer import OPDTrainer


# -----------------------
# Config (edit or use env)
# -----------------------
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "Qwen/Qwen2-0.5B-Instruct")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2-1.5B-Instruct")

DATASET_NAME = os.environ.get("DATASET_NAME", "gsm8k")
DATASET_CONFIG = os.environ.get("DATASET_CONFIG", "main")

# Keep small for sanity checks; increase later.
TRAIN_SAMPLES = int(os.environ.get("TRAIN_SAMPLES", "2000"))
EVAL_SAMPLES = int(os.environ.get("EVAL_SAMPLES", "256"))

# Training knobs
PER_DEVICE_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "800"))  # set -1 to use epochs instead
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", "1"))  # ignored if MAX_STEPS > 0
LR = float(os.environ.get("LR", "2e-6"))
OPD_MODE = os.environ.get("OPD_MODE", "entropy_baseline")  # "expectation" | "stochastic" | "entropy_baseline"
NUM_INNER_STEPS = int(os.environ.get("NUM_INNER_STEPS", "1"))   # L in Algorithm 4
REPLAY_BUFFER_SIZE = int(os.environ.get("REPLAY_BUFFER_SIZE", "1"))  # k in Algorithm 4


def _short_name(model_id: str) -> str:
    """Extract the repo name part from a HuggingFace model ID (e.g. 'meta-llama/Llama-3.1-8B-Instruct' -> 'Llama-3.1-8B-Instruct')."""
    return model_id.split("/")[-1]


# Auto-build OUTPUT_DIR from model names + OPD settings + timestamp when not explicitly set.
_default_output_dir = (
    f"opd_gsm8k"
    f"_S-{_short_name(STUDENT_MODEL)}"
    f"_T-{_short_name(TEACHER_MODEL)}"
    f"_{OPD_MODE}"
    f"_L{NUM_INNER_STEPS}"
    f"_buf{REPLAY_BUFFER_SIZE}"
    f"_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", _default_output_dir)

LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "100"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))

MAX_NEW_TOKENS_EVAL = int(os.environ.get("MAX_NEW_TOKENS_EVAL", "256"))
FLEXIBLE_EVAL = os.environ.get("FLEXIBLE_EVAL", "0") == "1"  # set to "1" to also accept $num$ format
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# GSM8K formatting helpers
# -----------------------
SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem and give the final answer.\n"
    "Put the final numeric answer on a line by itself in the format: #### <number>\n"
)

def gsm8k_to_messages(question: str, answer: Optional[str]) -> List[Dict[str, str]]:
    """
    TRL GKD examples use a list of messages with roles.
    If `answer` is provided, we include it as the assistant message (supervised target).
    """
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer.strip()})
    return msgs


_ANS_RE = re.compile(r"####\s*([\-]?\d[\d,\.]*)")
_ANS_RE_DOLLAR = re.compile(r"\$\s*([\-]?\d[\d,\.]*)\s*\$")

def extract_final_answer(text: str, flexible: bool = False) -> Optional[str]:
    """
    Parses the final numeric answer from a model response.

    1. GSM8K official format:   #### 42  (always accepted)
    2. Dollar-wrapped format:   $42$     (only if flexible=True)

    Commas are stripped for normalization (e.g. "1,000" -> "1000").
    """
    m = _ANS_RE.search(text)
    if m:
        return m.group(1).strip().replace(",", "")
    if flexible:
        m = _ANS_RE_DOLLAR.search(text)
        if m:
            return m.group(1).strip().replace(",", "")
    return None


def build_gsm8k_datasets(train_n: int, eval_n: int) -> Tuple[Dataset, Dataset]:
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
    train = ds["train"].select(range(min(train_n, len(ds["train"]))))
    test = ds["test"].select(range(min(eval_n, len(ds["test"]))))

    train_msgs = [gsm8k_to_messages(ex["question"], ex["answer"]) for ex in train]
    eval_msgs = [gsm8k_to_messages(ex["question"], ex["answer"]) for ex in test]

    train_dataset = Dataset.from_dict({"messages": train_msgs})
    eval_dataset = Dataset.from_dict({"messages": eval_msgs})
    return train_dataset, eval_dataset


# -----------------------
# Simple evaluation
# -----------------------
@torch.no_grad()
def gsm8k_exact_match(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_raw: Dataset,
    max_new_tokens: int = 256,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Greedy-generate answers, parse #### <num>, compare to gold parsed answer.
    This is a lightweight sanity eval (not a full harness).
    """
    model.eval()
    n = len(eval_raw) if limit is None else min(limit, len(eval_raw))
    correct = 0
    parsed = 0

    examples = []
    for i in range(n):
        q = eval_raw[i]["question"]
        gold = eval_raw[i]["answer"]
        gold_num = extract_final_answer(gold, flexible=FLEXIBLE_EVAL)

        msgs = gsm8k_to_messages(q, answer=None)

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        else:
            # Fallback: plain concatenation
            prompt = SYSTEM_PROMPT + "\nQ: " + q.strip() + "\nA:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0], skip_special_tokens=True)
        pred_num = extract_final_answer(gen, flexible=FLEXIBLE_EVAL)

        is_correct = (pred_num is not None) and (gold_num is not None) and (pred_num == gold_num)
        if pred_num is not None:
            parsed += 1
        if is_correct:
            correct += 1

        if i < 5:
            examples.append(
                {
                    "question": q,
                    "gold_num": gold_num,
                    "pred_num": pred_num,
                    "pred_excerpt": gen[-400:],
                }
            )

    acc = correct / n
    parse_rate = parsed / n
    return {
        "n": n,
        "exact_match": acc,
        "parse_rate": parse_rate,
        "correct": correct,
        "examples": examples,
    }


# -----------------------
# Plotting
# -----------------------
def plot_trainer_history(log_history: List[Dict[str, Any]], out_dir: str) -> None:
    """
    TRL/HF Trainer writes dicts containing keys like:
      {'loss':..., 'learning_rate':..., 'epoch':..., 'step':...}
      {'eval_loss':..., 'eval_runtime':..., 'epoch':..., 'step':...}
    We'll create a couple simple plots.
    """
    steps_train, loss_train = [], []
    steps_eval, loss_eval = [], []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry and "eval_loss" not in entry:
            steps_train.append(step)
            loss_train.append(entry["loss"])
        if "eval_loss" in entry:
            steps_eval.append(step)
            loss_eval.append(entry["eval_loss"])

    os.makedirs(out_dir, exist_ok=True)

    if steps_train:
        plt.figure()
        plt.plot(steps_train, loss_train)
        plt.xlabel("step")
        plt.ylabel("train loss")
        plt.title("GKD train loss")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "train_loss.png"))
        plt.close()

    if steps_eval:
        plt.figure()
        plt.plot(steps_eval, loss_eval)
        plt.xlabel("step")
        plt.ylabel("eval loss")
        plt.title("GKD eval loss")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "eval_loss.png"))
        plt.close()


def main() -> None:
    print(f"DEVICE={DEVICE}")
    print(f"STUDENT_MODEL={STUDENT_MODEL}")
    print(f"TEACHER_MODEL={TEACHER_MODEL}")
    print(f"TRAIN_SAMPLES={TRAIN_SAMPLES} EVAL_SAMPLES={EVAL_SAMPLES}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    print(f"OPD_MODE={OPD_MODE}")
    print(f"NUM_INNER_STEPS={NUM_INNER_STEPS}  REPLAY_BUFFER_SIZE={REPLAY_BUFFER_SIZE}")

    # Save run config so results are always self-describing
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_config = {
        "student_model": STUDENT_MODEL,
        "teacher_model": TEACHER_MODEL,
        "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
        "train_samples": TRAIN_SAMPLES,
        "eval_samples": EVAL_SAMPLES,
        "opd_mode": OPD_MODE,
        "num_inner_steps": NUM_INNER_STEPS,
        "replay_buffer_size": REPLAY_BUFFER_SIZE,
        "max_steps": MAX_STEPS,
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "output_dir": OUTPUT_DIR,
    }
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Load datasets
    train_dataset, eval_dataset = build_gsm8k_datasets(TRAIN_SAMPLES, EVAL_SAMPLES)

    # Also load raw eval split for exact-match evaluation
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG)
    eval_raw = raw["test"].select(range(min(EVAL_SAMPLES, len(raw["test"]))))

    # Tokenizer: use student tokenizer
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models
    # Tip: use bf16 on H200/H100/A100 for speed/memory.
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Pre-train eval
    pre = gsm8k_exact_match(model, tokenizer, eval_raw, max_new_tokens=MAX_NEW_TOKENS_EVAL, limit=64)
    print("Pre-train GSM8K (subset) exact match:", pre)

    # Training config
    args = GKDConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="no",
        do_train=True,
        do_eval=True,
        report_to=["wandb"],
        bf16=torch.cuda.is_available(),
        lmbda=1.0,
    )

    # Choose max_steps vs epochs
    if MAX_STEPS > 0:
        args.max_steps = MAX_STEPS
    else:
        args.num_train_epochs = NUM_EPOCHS

    trainer = OPDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        mode=OPD_MODE,
        num_inner_steps=NUM_INNER_STEPS,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
    )

    trainer.train()

    # Save final + plots
    trainer.save_model(OUTPUT_DIR)

    history = trainer.state.log_history
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "log_history.jsonl"), "w") as f:
        for e in history:
            f.write(json.dumps(e) + "\n")

    plot_trainer_history(history, OUTPUT_DIR)

    # Post-train eval
    post = gsm8k_exact_match(model, tokenizer, eval_raw, max_new_tokens=MAX_NEW_TOKENS_EVAL, limit=256)
    print("Post-train GSM8K exact match:", post)

    with open(os.path.join(OUTPUT_DIR, "eval_pre.json"), "w") as f:
        json.dump(pre, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "eval_post.json"), "w") as f:
        json.dump(post, f, indent=2)

    print(f"Done. Outputs in: {OUTPUT_DIR}")
    print(f"Plots: {OUTPUT_DIR}/train_loss.png and {OUTPUT_DIR}/eval_loss.png")


if __name__ == "__main__":
    main()
