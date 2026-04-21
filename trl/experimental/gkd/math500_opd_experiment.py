# math500_opd_experiment.py
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import matplotlib
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.gkd import GKDConfig
from trl.experimental.gkd.opd_trainer import OPDTrainer

try:
    from peft import LoraConfig
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


# -----------------------
# Config (edit or use env)
# -----------------------
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "Qwen/Qwen2-0.5B-Instruct")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2-7B-Instruct")

# Training data: full MATH train split; eval data: MATH-500 test split
TRAIN_DATASET_NAME = os.environ.get("TRAIN_DATASET_NAME", "EleutherAI/hendrycks_math")
TRAIN_DATASET_CONFIG = os.environ.get("TRAIN_DATASET_CONFIG", "all")
EVAL_DATASET_NAME = os.environ.get("EVAL_DATASET_NAME", "HuggingFaceH4/MATH-500")

TRAIN_SAMPLES = int(os.environ.get("TRAIN_SAMPLES", "7500"))
EVAL_SAMPLES = int(os.environ.get("EVAL_SAMPLES", "500"))

PER_DEVICE_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "800"))
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", "1"))
LR = float(os.environ.get("LR", "2e-6"))
OPD_MODE = os.environ.get("OPD_MODE", "entropy_baseline")
NUM_INNER_STEPS = int(os.environ.get("NUM_INNER_STEPS", "1"))
REPLAY_BUFFER_SIZE = int(os.environ.get("REPLAY_BUFFER_SIZE", "1"))
TRUST_REGION = os.environ.get("TRUST_REGION", "0") == "1"
PPO_CLIP_EPS = float(os.environ.get("PPO_CLIP_EPS", "0.2"))
USE_CORRECTION = os.environ.get("USE_CORRECTION", "0") == "1"
CORRECTION_ALPHA = float(os.environ.get("CORRECTION_ALPHA", "0.2"))
CORRECTION_LR = float(os.environ.get("CORRECTION_LR", "1e-3"))

USE_LORA = os.environ.get("USE_LORA", "0") == "1"
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
LORA_TARGET_MODULES = os.environ.get("LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj").split(",")


def _short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


_default_output_dir = (
    f"opd_math500"
    f"_S-{_short_name(STUDENT_MODEL)}"
    f"_T-{_short_name(TEACHER_MODEL)}"
    f"_{OPD_MODE}"
    f"{'_tr' if TRUST_REGION else ''}"
    f"{'_corr' if USE_CORRECTION else ''}"
    f"{'_lora' if USE_LORA else ''}"
    f"_L{NUM_INNER_STEPS}"
    f"_buf{REPLAY_BUFFER_SIZE}"
    f"_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", _default_output_dir)
HUB_MODEL_ID = f"QpiEImitation/opd_math500_S-{_short_name(STUDENT_MODEL)}_T-{_short_name(TEACHER_MODEL)}"

LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "100"))
MAX_NEW_TOKENS_EVAL = int(os.environ.get("MAX_NEW_TOKENS_EVAL", "512"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# MATH formatting helpers
# -----------------------
SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step.\n"
    "Put the final answer inside \\boxed{...}.\n"
)


def math_to_messages(problem: str, solution: Optional[str]) -> List[Dict[str, str]]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.strip()},
    ]
    if solution is not None:
        msgs.append({"role": "assistant", "content": solution.strip()})
    return msgs


# Handles one level of nested braces (sufficient for most MATH answers)
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_boxed_answer(text: str) -> Optional[str]:
    m = _BOXED_RE.search(text)
    return m.group(1).strip() if m else None


_HENDRYCKS_CONFIGS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def _load_train_dataset():
    if TRAIN_DATASET_CONFIG == "all" and TRAIN_DATASET_NAME == "EleutherAI/hendrycks_math":
        from datasets import concatenate_datasets
        splits = [load_dataset(TRAIN_DATASET_NAME, cfg)["train"] for cfg in _HENDRYCKS_CONFIGS]
        return concatenate_datasets(splits)
    return load_dataset(TRAIN_DATASET_NAME, TRAIN_DATASET_CONFIG)["train"]


def build_math_datasets(train_n: int, eval_n: int) -> Tuple[Dataset, Dataset]:
    train_split = _load_train_dataset()
    train = train_split.select(range(min(train_n, len(train_split))))
    train_msgs = [math_to_messages(ex["problem"], ex["solution"]) for ex in train]
    train_dataset = Dataset.from_dict({"messages": train_msgs})

    eval_ds = load_dataset(EVAL_DATASET_NAME)
    test = eval_ds["test"].select(range(min(eval_n, len(eval_ds["test"]))))
    eval_msgs = [math_to_messages(ex["problem"], ex["solution"]) for ex in test]
    eval_dataset = Dataset.from_dict({"messages": eval_msgs})

    return train_dataset, eval_dataset


# -----------------------
# Evaluation
# -----------------------
@torch.no_grad()
def math500_exact_match(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_raw,
    max_new_tokens: int = 512,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    n = len(eval_raw) if limit is None else min(limit, len(eval_raw))
    correct = 0
    parsed = 0
    examples = []

    for i in range(n):
        problem = eval_raw[i]["problem"]
        gold = eval_raw[i]["answer"]

        msgs = math_to_messages(problem, solution=None)

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        else:
            prompt = SYSTEM_PROMPT + "\n" + problem.strip()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_boxed_answer(gen)

        is_correct = pred is not None and gold is not None and pred.strip() == gold.strip()
        if pred is not None:
            parsed += 1
        if is_correct:
            correct += 1

        if i < 5:
            examples.append({"problem": problem, "gold": gold, "pred": pred, "pred_excerpt": gen[-400:]})

    return {
        "n": n,
        "exact_match": correct / n,
        "parse_rate": parsed / n,
        "correct": correct,
        "examples": examples,
    }


# -----------------------
# Plotting
# -----------------------
def plot_trainer_history(log_history: List[Dict[str, Any]], out_dir: str) -> None:
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
        plt.xlabel("step"); plt.ylabel("train loss"); plt.title("OPD train loss")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "train_loss.png")); plt.close()
    if steps_eval:
        plt.figure()
        plt.plot(steps_eval, loss_eval)
        plt.xlabel("step"); plt.ylabel("eval loss"); plt.title("OPD eval loss")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "eval_loss.png")); plt.close()


def main() -> None:
    print(f"DEVICE={DEVICE}")
    print(f"STUDENT_MODEL={STUDENT_MODEL}")
    print(f"TEACHER_MODEL={TEACHER_MODEL}")
    print(f"TRAIN_SAMPLES={TRAIN_SAMPLES} EVAL_SAMPLES={EVAL_SAMPLES}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    print(f"OPD_MODE={OPD_MODE}")
    print(f"TRUST_REGION={TRUST_REGION}  PPO_CLIP_EPS={PPO_CLIP_EPS}")
    print(f"NUM_INNER_STEPS={NUM_INNER_STEPS}  REPLAY_BUFFER_SIZE={REPLAY_BUFFER_SIZE}")
    print(f"USE_CORRECTION={USE_CORRECTION}  CORRECTION_ALPHA={CORRECTION_ALPHA}  CORRECTION_LR={CORRECTION_LR}")
    print(f"USE_LORA={USE_LORA}" + (f"  r={LORA_R}  alpha={LORA_ALPHA}  dropout={LORA_DROPOUT}" if USE_LORA else ""))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_config = {
        "method": "opd",
        "student_model": STUDENT_MODEL,
        "teacher_model": TEACHER_MODEL,
        "train_dataset": f"{TRAIN_DATASET_NAME}/{TRAIN_DATASET_CONFIG}",
        "eval_dataset": EVAL_DATASET_NAME,
        "train_samples": TRAIN_SAMPLES,
        "eval_samples": EVAL_SAMPLES,
        "opd_mode": OPD_MODE,
        "trust_region": TRUST_REGION,
        "ppo_clip_eps": PPO_CLIP_EPS,
        "use_correction": USE_CORRECTION,
        "correction_alpha": CORRECTION_ALPHA,
        "correction_lr": CORRECTION_LR,
        "num_inner_steps": NUM_INNER_STEPS,
        "replay_buffer_size": REPLAY_BUFFER_SIZE,
        "max_steps": MAX_STEPS,
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "output_dir": OUTPUT_DIR,
        "use_lora": USE_LORA,
        "lora_r": LORA_R if USE_LORA else None,
        "lora_alpha": LORA_ALPHA if USE_LORA else None,
        "lora_dropout": LORA_DROPOUT if USE_LORA else None,
        "lora_target_modules": LORA_TARGET_MODULES if USE_LORA else None,
    }
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    train_dataset, eval_dataset = build_math_datasets(TRAIN_SAMPLES, EVAL_SAMPLES)

    eval_ds = load_dataset(EVAL_DATASET_NAME)
    eval_raw = eval_ds["test"].select(range(min(EVAL_SAMPLES, len(eval_ds["test"]))))

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    pre = math500_exact_match(model, tokenizer, eval_raw, max_new_tokens=MAX_NEW_TOKENS_EVAL, limit=64)
    print("Pre-train MATH500 (subset) exact match:", pre)

    args = GKDConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=LOGGING_STEPS,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=200,
        do_train=True,
        do_eval=False,
        report_to=["wandb"],
        bf16=torch.cuda.is_available(),
        lmbda=1.0,
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
    )

    if MAX_STEPS > 0:
        args.max_steps = MAX_STEPS
    else:
        args.num_train_epochs = NUM_EPOCHS

    peft_config = None
    if USE_LORA:
        if not _PEFT_AVAILABLE:
            raise ImportError("USE_LORA=1 but peft is not installed. Run: pip install peft")
        peft_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
        )

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
        trust_region=TRUST_REGION,
        ppo_clip_eps=PPO_CLIP_EPS,
        use_correction=USE_CORRECTION,
        correction_alpha=CORRECTION_ALPHA,
        correction_lr=CORRECTION_LR,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    history = trainer.state.log_history
    with open(os.path.join(OUTPUT_DIR, "log_history.jsonl"), "w") as f:
        for e in history:
            f.write(json.dumps(e) + "\n")

    plot_trainer_history(history, OUTPUT_DIR)

    post = math500_exact_match(model, tokenizer, eval_raw, max_new_tokens=MAX_NEW_TOKENS_EVAL)
    print("Post-train MATH500 exact match:", post)

    with open(os.path.join(OUTPUT_DIR, "eval_pre.json"), "w") as f:
        json.dump(pre, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "eval_post.json"), "w") as f:
        json.dump(post, f, indent=2)

    print(f"Done. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
