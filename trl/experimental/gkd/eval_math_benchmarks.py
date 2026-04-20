#!/usr/bin/env python3
"""
Evaluate a model on three math benchmarks:
  - GSM8K   (1319 test problems, grade-school arithmetic)
  - MATH500 (500 competition problems, lighteval/MATH-Hard)
  - SVAMP   (1000 adversarial arithmetic problems, ChilleD/SVAMP)

Results are saved as eval_gsm8k.json, eval_math500.json, eval_svamp.json
inside --output_dir (defaults to --model_dir).

Usage:
  python eval_math_benchmarks.py --model_dir path/to/model
  python eval_math_benchmarks.py --model_dir path/to/model --benchmarks gsm8k math500
"""

import argparse
import json
import os
import re
from fractions import Fraction
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Answer parsing ────────────────────────────────────────────────────────────

_HASH_RE   = re.compile(r"####\s*([\-]?\d[\d,\.]*)")
_BOXED_RE  = re.compile(r"\\boxed\{([^}]*)\}")
_DOLLAR_RE = re.compile(r"\$\s*([\-]?\d[\d,\.]*)\s*\$")
_NUM_RE    = re.compile(r"[\-]?\d[\d,\.]*")


def _strip(s: str) -> str:
    return s.strip().replace(",", "").replace(" ", "")


def _to_number(s: str) -> Optional[float]:
    s = _strip(s)
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    return None


def parse_gsm8k(text: str) -> Optional[str]:
    m = _HASH_RE.search(text)
    if m:
        return _strip(m.group(1))
    m = _BOXED_RE.search(text)
    if m:
        return _strip(m.group(1))
    m = _DOLLAR_RE.search(text)
    if m:
        return _strip(m.group(1))
    return None


def parse_math500(text: str) -> Optional[str]:
    """Extract answer from model output for MATH-style problems."""
    m = _BOXED_RE.search(text)
    if m:
        return _strip(m.group(1))
    m = _HASH_RE.search(text)
    if m:
        return _strip(m.group(1))
    return None


def math500_equiv(pred: str, gold: str) -> bool:
    """Numeric equivalence when possible, else normalised string match."""
    p, g = _strip(pred), _strip(gold)
    if p == g:
        return True
    pn, gn = _to_number(p), _to_number(g)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    return False


def parse_svamp(text: str) -> Optional[float]:
    m = _HASH_RE.search(text)
    if m:
        n = _to_number(m.group(1))
        if n is not None:
            return n
    m = _BOXED_RE.search(text)
    if m:
        n = _to_number(m.group(1))
        if n is not None:
            return n
    # Last numeric in output
    nums = _NUM_RE.findall(text.split("Q:")[0] if "Q:" in text else text)
    if nums:
        n = _to_number(nums[-1])
        return n
    return None


# ── Prompt helpers ────────────────────────────────────────────────────────────

MATH_SYSTEM = (
    "You are an expert mathematician. Solve the problem step by step.\n"
    "Put your final answer inside \\boxed{} at the end.\n"
)

GSM8K_SYSTEM = (
    "You are a helpful math tutor. Solve the problem and give the final answer.\n"
    "Put the final numeric answer on a line by itself in the format: #### <number>\n"
)


def make_prompt(tokenizer, system: str, question: str) -> str:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user",   "content": question.strip()},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return system + "\nQ: " + question.strip() + "\nA:"


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ── Benchmark evaluators ──────────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, n: int, max_new_tokens: int) -> dict:
    ds = load_dataset("gsm8k", "main", split="test")
    n = min(n, len(ds))
    correct = parsed = 0
    for i in range(n):
        gold = parse_gsm8k(ds[i]["answer"])
        resp = generate(model, tokenizer,
                        make_prompt(tokenizer, GSM8K_SYSTEM, ds[i]["question"]),
                        max_new_tokens)
        pred = parse_gsm8k(resp)
        if pred is not None:
            parsed += 1
        if pred is not None and gold is not None and pred == gold:
            correct += 1
        if (i + 1) % 100 == 0:
            print(f"  GSM8K [{i+1}/{n}]  acc={correct/(i+1):.3f}")
    return {"n": n, "exact_match": correct / n, "parse_rate": parsed / n,
            "correct": correct}


def eval_math500(model, tokenizer, max_new_tokens: int) -> dict:
    ds = load_dataset("lighteval/MATH-Hard", split="test")
    n = len(ds)
    correct = parsed = 0
    for i in range(n):
        gold_raw = ds[i].get("answer") or parse_math500(ds[i].get("solution", ""))
        if gold_raw is None:
            continue
        gold = _strip(gold_raw)
        problem = ds[i].get("problem") or ds[i].get("question", "")
        resp = generate(model, tokenizer,
                        make_prompt(tokenizer, MATH_SYSTEM, problem),
                        max_new_tokens)
        pred = parse_math500(resp)
        if pred is not None:
            parsed += 1
        if pred is not None and math500_equiv(pred, gold):
            correct += 1
        if (i + 1) % 100 == 0:
            print(f"  MATH500 [{i+1}/{n}]  acc={correct/(i+1):.3f}")
    return {"n": n, "exact_match": correct / n, "parse_rate": parsed / n,
            "correct": correct}


def eval_svamp(model, tokenizer, max_new_tokens: int) -> dict:
    ds = load_dataset("ChilleD/SVAMP", split="test")
    n = len(ds)
    correct = parsed = 0
    for i in range(n):
        gold = float(ds[i]["Answer"])
        body = ds[i].get("Body", "") + " " + ds[i].get("Question", "")
        resp = generate(model, tokenizer,
                        make_prompt(tokenizer, GSM8K_SYSTEM, body.strip()),
                        max_new_tokens)
        pred = parse_svamp(resp)
        if pred is not None:
            parsed += 1
        if pred is not None and abs(pred - gold) < 1e-6:
            correct += 1
        if (i + 1) % 100 == 0:
            print(f"  SVAMP [{i+1}/{n}]  acc={correct/(i+1):.3f}")
    return {"n": n, "exact_match": correct / n, "parse_rate": parsed / n,
            "correct": correct}


# ── Main ──────────────────────────────────────────────────────────────────────

BENCHMARK_NAMES = ["gsm8k", "math500", "svamp"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True,
                        help="Local model directory or HuggingFace model ID")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save JSON results (defaults to --model_dir)")
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARK_NAMES,
                        choices=BENCHMARK_NAMES,
                        help="Which benchmarks to run (default: all three)")
    parser.add_argument("--gsm8k_n", type=int, default=1319)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    out_dir = args.output_dir or (
        args.model_dir if os.path.isdir(args.model_dir)
        else "eval_" + args.model_dir.replace("/", "_")
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model      : {args.model_dir}")
    print(f"Output dir : {out_dir}")
    print(f"Benchmarks : {args.benchmarks}")

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    summary = {}

    if "gsm8k" in args.benchmarks:
        print("\n── GSM8K ──")
        res = eval_gsm8k(model, tokenizer, args.gsm8k_n, args.max_new_tokens)
        json.dump(res, open(os.path.join(out_dir, "eval_gsm8k.json"), "w"), indent=2)
        print(f"  GSM8K   acc={res['exact_match']:.4f}  ({res['correct']}/{res['n']})")
        summary["gsm8k"] = res["exact_match"]

    if "math500" in args.benchmarks:
        print("\n── MATH500 ──")
        res = eval_math500(model, tokenizer, args.max_new_tokens)
        json.dump(res, open(os.path.join(out_dir, "eval_math500.json"), "w"), indent=2)
        print(f"  MATH500 acc={res['exact_match']:.4f}  ({res['correct']}/{res['n']})")
        summary["math500"] = res["exact_match"]

    if "svamp" in args.benchmarks:
        print("\n── SVAMP ──")
        res = eval_svamp(model, tokenizer, args.max_new_tokens)
        json.dump(res, open(os.path.join(out_dir, "eval_svamp.json"), "w"), indent=2)
        print(f"  SVAMP   acc={res['exact_match']:.4f}  ({res['correct']}/{res['n']})")
        summary["svamp"] = res["exact_match"]

    json.dump(summary, open(os.path.join(out_dir, "eval_summary.json"), "w"), indent=2)
    print(f"\nSummary saved to {out_dir}/eval_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
