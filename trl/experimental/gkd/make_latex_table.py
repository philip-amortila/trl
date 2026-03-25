#!/usr/bin/env python3
"""Generate a LaTeX table comparing algorithms with/without trust region."""

import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))


def load_acc(folder):
    path = os.path.join(BASE, folder, "eval_gsm8k.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d["exact_match_flex"]


# Map: (algorithm_label, trust_region) -> folder
EXPERIMENTS = {
    ("GKD",             False): "gkd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_20260324_113250",
    ("GKD",             True):  None,  # no TR variant provided
    ("OPD-Softmax",     False): "opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_stochastic_L10_buf10_20260324_113250",
    ("OPD-Softmax",     True):  "opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_stochastic_tr_L10_buf10_20260324_113250",
    ("OPD-Entropy",     False): "opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_entropy_baseline_L10_buf10_20260324_113251",
    ("OPD-Entropy",     True):  "opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_entropy_baseline_tr_L10_buf10_20260324_113251",
    ("OPD-Expectation", False): "opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_expectation_L10_buf10_20260324_113251",
    ("OPD-Expectation", True):  "opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct_expectation_tr_L10_buf10_20260324_113250",
}

ALGORITHMS = ["GKD", "OPD-Softmax", "OPD-Entropy", "OPD-Expectation"]

results = {}
for (algo, tr), folder in EXPERIMENTS.items():
    if folder is None:
        results[(algo, tr)] = None
    else:
        acc = load_acc(folder)
        results[(algo, tr)] = acc
        status = f"{acc:.4f}" if acc is not None else "MISSING"
        print(f"  {algo:20s}  TR={tr}  →  {status}")


def fmt(val):
    if val is None:
        return "---"
    return f"{val * 100:.1f}"


latex = r"""\begin{table}[h]
\centering
\caption{GSM8K accuracy (exact\_match\_flex \%) for student Qwen2-0.5B-Instruct
         distilled from Qwen2-1.5B-Instruct.}
\label{tab:gsm8k_results}
\begin{tabular}{lcc}
\toprule
\textbf{Algorithm} & \textbf{No Trust Region} & \textbf{Trust Region} \\
\midrule
"""

for algo in ALGORITHMS:
    no_tr = fmt(results[(algo, False)])
    with_tr = fmt(results[(algo, True)])
    latex += f"{algo} & {no_tr} & {with_tr} \\\\\n"

latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

print("\n" + "=" * 60)
print(latex)

out_path = os.path.join(BASE, "results_table.tex")
with open(out_path, "w") as f:
    f.write(latex)
print(f"Saved to {out_path}")
