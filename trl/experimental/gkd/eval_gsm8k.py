"""
Standalone GSM8K evaluation script.

Loads a saved model and evaluates exact-match accuracy on GSM8K test set.

Always reports both strict (#### only) and flexible (\boxed{} / $num$) exact match.

Usage:
  python eval_gsm8k.py --model_dir gkd_gsm8k_out
  python eval_gsm8k.py --model_dir opd_gsm8k_out_entropy_baseline --n 1319

Or via env vars:
  MODEL_DIR=gkd_gsm8k_out python eval_gsm8k.py
"""
import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Answer parsing ────────────────────────────────────────────────────────────

_ANS_RE = re.compile(r"####\s*([\-]?\d[\d,\.]*)")
_ANS_RE_BOXED = re.compile(r"\\boxed\{([\-]?\d[\d,\.]*)\}")
_ANS_RE_DOLLAR = re.compile(r"\$\s*([\-]?\d[\d,\.]*)\s*\$")

def extract_final_answer(text: str, flexible: bool = False) -> Optional[str]:
    m = _ANS_RE.search(text)
    if m:
        return m.group(1).strip().replace(",", "")
    if flexible:
        m = _ANS_RE_BOXED.search(text)
        if m:
            return m.group(1).strip().replace(",", "")
        m = _ANS_RE_DOLLAR.search(text)
        if m:
            return m.group(1).strip().replace(",", "")
    return None


# ── Prompt formatting ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem and give the final answer.\n"
    "Put the final numeric answer on a line by itself in the format: #### <number>\n"
)

def make_prompt(tokenizer, question: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return SYSTEM_PROMPT + "\nQ: " + question.strip() + "\nA:"


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tokenizer, n: int, max_new_tokens: int) -> Dict[str, Any]:
    ds = load_dataset("gsm8k", "main", split="test")
    n = min(n, len(ds))

    model.eval()
    correct_strict, correct_flex = 0, 0
    parsed_strict, parsed_flex = 0, 0
    examples = []

    for i in range(n):
        question = ds[i]["question"]
        gold_strict = extract_final_answer(ds[i]["answer"], flexible=False)
        gold_flex   = extract_final_answer(ds[i]["answer"], flexible=True)

        prompt = make_prompt(tokenizer, question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        pred_strict = extract_final_answer(response, flexible=False)
        pred_flex   = extract_final_answer(response, flexible=True)

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
                "pred_excerpt": response[-400:],
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=os.environ.get("MODEL_DIR", "gkd_gsm8k_out"),
                        help="Path to saved model directory, or a HuggingFace model ID (e.g. Qwen/Qwen2-1.5B-Instruct)")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save eval_gsm8k.json (defaults to --model_dir if it's a local path, "
                             "otherwise a sanitized version of the model ID)")
    parser.add_argument("--n", type=int, default=int(os.environ.get("EVAL_N", "1319")),
                        help="Number of test examples to evaluate (default: full test set of 1319)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is not None:
        out_dir = args.output_dir
    elif os.path.isdir(args.model_dir):
        out_dir = args.model_dir
    else:
        # HuggingFace model ID like "Qwen/Qwen2-1.5B-Instruct" → "eval_Qwen2-1.5B-Instruct"
        out_dir = "eval_" + args.model_dir.replace("/", "_")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model: {args.model_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Eval n: {args.n}")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    results = evaluate(model, tokenizer, n=args.n, max_new_tokens=args.max_new_tokens)

    print(f"\n── Results for {args.model_dir} ──")
    print(f"  exact_match (strict) : {results['exact_match_strict']:.3f}  ({results['correct_strict']}/{results['n']})")
    print(f"  exact_match (flex)   : {results['exact_match_flex']:.3f}  ({results['correct_flex']}/{results['n']})")
    print(f"  parse_rate  (strict) : {results['parse_rate_strict']:.3f}")
    print(f"  parse_rate  (flex)   : {results['parse_rate_flex']:.3f}")

    out_path = os.path.join(out_dir, "eval_gsm8k.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
