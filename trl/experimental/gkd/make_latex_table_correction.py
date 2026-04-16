#!/usr/bin/env python3
"""
Generate a LaTeX table combining trust-region and correction-network results.

Columns: No Corr / No TR | No Corr / TR | Corr / No TR | Corr + TR
Rows:    GKD | OPD-Entropy | OPD-Expectation | OPD-Stochastic

Directories are auto-discovered: the most recent folder that contains
eval_gsm8k.json is selected for each (algorithm, tr, corr) combination.
"""

import json
import os
from pathlib import Path

BASE = Path(__file__).parent.resolve()


def find_latest_with_eval(pattern: str) -> Path | None:
    """Return the most recently modified dir matching pattern that has eval_gsm8k.json."""
    candidates = [
        d for d in sorted(BASE.glob(pattern), key=lambda d: d.stat().st_mtime, reverse=True)
        if (d / "eval_gsm8k.json").exists()
    ]
    return candidates[0] if candidates else None


def load_acc(folder: Path | None) -> float | None:
    if folder is None:
        return None
    path = folder / "eval_gsm8k.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["exact_match_flex"]


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

BASE_PAT = "opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct"
GKD_PAT  = "gkd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct"

DIRS = {
    # (algo_label, tr, corr) -> glob pattern
    ("GKD",             False, False): f"{GKD_PAT}_*",
    ("GKD",             True,  False): None,   # no TR variant for GKD
    ("GKD",             False, True):  None,   # no correction variant for GKD
    ("GKD",             True,  True):  None,

    ("OPD-Entropy",     False, False): f"{BASE_PAT}_entropy_baseline_L*_buf*",
    ("OPD-Entropy",     True,  False): f"{BASE_PAT}_entropy_baseline_tr_L*_buf*",
    ("OPD-Entropy",     False, True):  f"{BASE_PAT}_entropy_baseline_corr_L*_buf*",
    ("OPD-Entropy",     True,  True):  f"{BASE_PAT}_entropy_baseline_tr_corr_L*_buf*",

    ("OPD-Expectation", False, False): f"{BASE_PAT}_expectation_L*_buf*",
    ("OPD-Expectation", True,  False): f"{BASE_PAT}_expectation_tr_L*_buf*",
    ("OPD-Expectation", False, True):  f"{BASE_PAT}_expectation_corr_L*_buf*",
    ("OPD-Expectation", True,  True):  f"{BASE_PAT}_expectation_tr_corr_L*_buf*",

    ("OPD-Stochastic",  False, False): f"{BASE_PAT}_stochastic_L*_buf*",
    ("OPD-Stochastic",  True,  False): f"{BASE_PAT}_stochastic_tr_L*_buf*",
    ("OPD-Stochastic",  False, True):  f"{BASE_PAT}_stochastic_corr_L*_buf*",
    ("OPD-Stochastic",  True,  True):  f"{BASE_PAT}_stochastic_tr_corr_L*_buf*",
}

results = {}
print("Discovered model directories:")
for (algo, tr, corr), pattern in DIRS.items():
    if pattern is None:
        results[(algo, tr, corr)] = None
        print(f"  {algo:20s}  TR={int(tr)}  Corr={int(corr)}  →  ---")
        continue
    folder = find_latest_with_eval(pattern)
    acc = load_acc(folder)
    results[(algo, tr, corr)] = acc
    name = folder.name if folder else "NOT FOUND"
    status = f"{acc:.4f}" if acc is not None else "MISSING eval"
    print(f"  {algo:20s}  TR={int(tr)}  Corr={int(corr)}  →  {status}  [{name}]")


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

ALGORITHMS = ["GKD", "OPD-Entropy", "OPD-Expectation", "OPD-Stochastic"]


def fmt(val: float | None) -> str:
    if val is None:
        return "---"
    return f"{val * 100:.1f}"


latex = r"""\begin{table}[h]
\centering
\caption{GSM8K accuracy (exact\_match\_flex \%) for student Qwen2-0.5B-Instruct
         distilled from Qwen2-1.5B-Instruct.
         Correction refers to the $\zeta$-network (Algorithm~8);
         TR is the PPO-style trust-region constraint.}
\label{tab:gsm8k_correction}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{\textbf{No Correction}} & \multicolumn{2}{c}{\textbf{With Correction}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
\textbf{Algorithm} & \textbf{No TR} & \textbf{TR} & \textbf{No TR} & \textbf{TR} \\
\midrule
"""

for algo in ALGORITHMS:
    cols = [
        fmt(results[(algo, False, False)]),
        fmt(results[(algo, True,  False)]),
        fmt(results[(algo, False, True)]),
        fmt(results[(algo, True,  True)]),
    ]
    latex += f"{algo} & {cols[0]} & {cols[1]} & {cols[2]} & {cols[3]} \\\\\n"

latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

print("\n" + "=" * 70)
print(latex)

out_path = BASE / "results_table_correction.tex"
out_path.write_text(latex)
print(f"Saved to {out_path}")
