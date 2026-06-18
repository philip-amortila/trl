"""
GSM8K evaluation for seq2seq T5/Flan-T5 models.

Counterpart of eval_gsm8k.py for encoder-decoder models trained with
gkd_t5_experiment.py or opd_t5_experiment.py.

Usage:
  python eval_gsm8k_t5.py --model_dir gkd_gsm8k_S-flan-t5-small_T-flan-t5-xl_*
  python eval_gsm8k_t5.py --model_dir google/flan-t5-small   # baseline
  python eval_gsm8k_t5.py --model_dir gkd_gsm8k_S-flan-t5-small_T-flan-t5-xl_* --teacher_model_dir google/flan-t5-xl
"""
import argparse
import json
import os
import re
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


_ANS_RE = re.compile(r"####\s*([\-]?\d[\d,\.]*)")
_ANS_RE_BOXED = re.compile(r"\\boxed\{([\-]?\d[\d,\.]*)\}")
_ANS_RE_DOLLAR = re.compile(r"\$\s*([\-]?\d[\d,\.]*)\s*\$")
_ANS_RE_FINAL = re.compile(r"[Tt]he\s+final\s+answer\s*:\s*\$?\s*([\-]?\d[\d,]*(?:\.\d+)?)")

def extract_answer(text: str, flexible: bool = False) -> Optional[str]:
    m = _ANS_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    if flexible:
        m = _ANS_RE_FINAL.search(text)
        if m:
            return m.group(1).replace(",", "")
        m = _ANS_RE_BOXED.search(text)
        if m:
            return m.group(1).replace(",", "")
        m = _ANS_RE_DOLLAR.search(text)
        if m:
            return m.group(1).replace(",", "")
    return None


INPUT_PREFIX = (
    "Solve the math problem step by step. "
    "Write your final answer as #### <number>.\n\n"
)

def format_input(question: str) -> str:
    return INPUT_PREFIX + question.strip()


@torch.no_grad()
def evaluate(model, tokenizer, n: int, max_new_tokens: int) -> Dict[str, Any]:
    ds = load_dataset("gsm8k", "main", split="test")
    n = min(n, len(ds))

    model.eval()
    correct_strict = correct_flex = parsed_strict = parsed_flex = 0
    examples = []

    for i in range(n):
        question = ds[i]["question"]
        gold_strict = extract_answer(ds[i]["answer"], flexible=False)
        gold_flex   = extract_answer(ds[i]["answer"], flexible=True)

        enc = tokenizer(
            format_input(question),
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        pred_strict = extract_answer(response, flexible=False)
        pred_flex   = extract_answer(response, flexible=True)

        if pred_strict is not None:
            parsed_strict += 1
        if pred_strict is not None and gold_strict is not None and pred_strict == gold_strict:
            correct_strict += 1

        if pred_flex is not None:
            parsed_flex += 1
        if pred_flex is not None and gold_flex is not None and pred_flex == gold_flex:
            correct_flex += 1

        if i < 10:
            examples.append({
                "question": question,
                "gold_num": gold_flex,
                "pred_strict": pred_strict,
                "pred_flex": pred_flex,
                "pred_text": response,
            })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}]  strict={correct_strict/(i+1):.3f}  flex={correct_flex/(i+1):.3f}  parse_strict={parsed_strict/(i+1):.3f}  parse_flex={parsed_flex/(i+1):.3f}")

    return {
        "n": n,
        "exact_match_strict": correct_strict / n,
        "exact_match_flex":   correct_flex / n,
        "parse_rate_strict":  parsed_strict / n,
        "parse_rate_flex":    parsed_flex / n,
        "correct_strict": correct_strict,
        "correct_flex":   correct_flex,
        "examples": examples,
    }


def load_model(model_id: str, torch_dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return model, tokenizer


def print_results(label: str, results: dict) -> None:
    print(f"\n── Results for {label} ──")
    print(f"  exact_match (strict) : {results['exact_match_strict']:.3f}  ({results['correct_strict']}/{results['n']})")
    print(f"  exact_match (flex)   : {results['exact_match_flex']:.3f}  ({results['correct_flex']}/{results['n']})")
    print(f"  parse_rate  (strict) : {results['parse_rate_strict']:.3f}")
    print(f"  parse_rate  (flex)   : {results['parse_rate_flex']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=os.environ.get("MODEL_DIR", ""),
        help="Path to a saved T5 model directory, or a HuggingFace model ID",
    )
    parser.add_argument(
        "--teacher_model_dir",
        default=os.environ.get("TEACHER_MODEL_DIR", ""),
        help="Optional teacher model to evaluate alongside the student",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n", type=int, default=int(os.environ.get("EVAL_N", "1319")),
                        help="Number of test examples (default: full 1319)")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    if not args.model_dir:
        parser.error("--model_dir is required")

    if args.output_dir is not None:
        out_dir = args.output_dir
    elif os.path.isdir(args.model_dir):
        out_dir = args.model_dir
    else:
        out_dir = "eval_" + args.model_dir.replace("/", "_")
    os.makedirs(out_dir, exist_ok=True)

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Student:    {args.model_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Eval n:     {args.n}")

    model, tokenizer = load_model(args.model_dir, torch_dtype)
    results = evaluate(model, tokenizer, n=args.n, max_new_tokens=args.max_new_tokens)
    print_results(args.model_dir, results)

    out_path = os.path.join(out_dir, "eval_gsm8k_t5.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to: {out_path}")

    if args.teacher_model_dir:
        print(f"\nTeacher:    {args.teacher_model_dir}")
        del model  # free GPU memory before loading teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        teacher_model, teacher_tokenizer = load_model(args.teacher_model_dir, torch_dtype)
        teacher_results = evaluate(teacher_model, teacher_tokenizer, n=args.n, max_new_tokens=args.max_new_tokens)
        print_results(args.teacher_model_dir, teacher_results)

        teacher_out_path = os.path.join(out_dir, "eval_gsm8k_t5_teacher.json")
        with open(teacher_out_path, "w") as f:
            json.dump(teacher_results, f, indent=2)
        print(f"  Saved to: {teacher_out_path}")


if __name__ == "__main__":
    main()
