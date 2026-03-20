"""
Plot training curves and eval results for GKD, entropy_baseline, expectation.
Usage: python plot_results.py
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

RUNS = {
    "OPD softmax":           "opd_gsm8k_out_softmax_20260317_113153",
    "OPD softmax L10 buf10": "opd_gsm8k_out_softmax_L10_buf10_20260319_163546",
    "OPD softmax L10 buf100":"opd_gsm8k_out_softmax_L10_buf100_20260319_164428",
}
COLORS = {
    "OPD softmax":           "#1f77b4",
    "OPD softmax L10 buf10": "#ff7f0e",
    "OPD softmax L10 buf100":"#2ca02c",
}

def load_log(run_dir):
    path = os.path.join(BASE, run_dir, "log_history.jsonl")
    train, eval_ = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "loss" in d:
                train.append(d)
            elif "eval_loss" in d:
                eval_.append(d)
    return train, eval_

def load_eval_results(run_dir):
    path = os.path.join(BASE, run_dir, "eval_gsm8k.json")
    with open(path) as f:
        return json.load(f)

# ── layout ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

ax_train = fig.add_subplot(gs[0, 0])
ax_eval  = fig.add_subplot(gs[0, 1])
ax_grad  = fig.add_subplot(gs[0, 2])
ax_em    = fig.add_subplot(gs[1, 0])
ax_parse = fig.add_subplot(gs[1, 1])
ax_table = fig.add_subplot(gs[1, 2])
ax_table.axis("off")

labels = list(RUNS.keys())
x      = np.arange(len(labels))
w      = 0.35
colors = [COLORS[l] for l in labels]

em_strict, em_flex, pr_strict, pr_flex = {}, {}, {}, {}

for label, run_dir in RUNS.items():
    color = COLORS[label]
    train, eval_ = load_log(run_dir)

    steps_t = [d["step"] for d in train]
    loss_t  = [d["loss"] for d in train]
    grad_t  = [d["grad_norm"] for d in train]
    steps_e = [d["step"]     for d in eval_]
    loss_e  = [d["eval_loss"] for d in eval_]

    ax_train.plot(steps_t, loss_t, color=color, alpha=0.8, label=label)
    ax_eval.plot(steps_e, loss_e, color=color, marker="o", ms=4, label=label)
    ax_grad.plot(steps_t, grad_t, color=color, alpha=0.6, label=label)

    res = load_eval_results(run_dir)
    em_strict[label] = res["exact_match_strict"]
    em_flex[label]   = res["exact_match_flex"]
    pr_strict[label] = res["parse_rate_strict"]
    pr_flex[label]   = res["parse_rate_flex"]

ax_train.set_title("Training Loss")
ax_train.set_xlabel("Step"); ax_train.set_ylabel("Loss")
ax_train.legend(fontsize=8)

ax_eval.set_title("Validation Loss (cross-entropy)")
ax_eval.set_xlabel("Step"); ax_eval.set_ylabel("Eval Loss")
ax_eval.legend(fontsize=8)

ax_grad.set_title("Gradient Norm")
ax_grad.set_xlabel("Step"); ax_grad.set_ylabel("Grad Norm")
ax_grad.legend(fontsize=8)

# Exact match: strict vs flexible side-by-side
bars_s = ax_em.bar(x - w/2, [em_strict[l] for l in labels], w,
                   color=colors, alpha=0.4, label="strict (####)")
bars_f = ax_em.bar(x + w/2, [em_flex[l]   for l in labels], w,
                   color=colors, alpha=1.0, label=r"flexible (\boxed{}/$n$)")
ax_em.set_title(f"Exact Match (n=1319)")
ax_em.set_xticks(x); ax_em.set_xticklabels(labels, fontsize=8)
ax_em.set_ylabel("Exact Match"); ax_em.set_ylim(0, 0.6)
ax_em.legend(fontsize=7)
for bar in list(bars_s) + list(bars_f):
    h = bar.get_height()
    if h > 0:
        ax_em.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                   f"{h:.3f}", ha="center", va="bottom", fontsize=7)

# Parse rate
bars_ps = ax_parse.bar(x - w/2, [pr_strict[l] for l in labels], w,
                       color=colors, alpha=0.4, label="strict")
bars_pf = ax_parse.bar(x + w/2, [pr_flex[l]   for l in labels], w,
                       color=colors, alpha=1.0, label="flexible")
ax_parse.set_title(f"Parse Rate (n=1319)")
ax_parse.set_xticks(x); ax_parse.set_xticklabels(labels, fontsize=8)
ax_parse.set_ylabel("Parse Rate"); ax_parse.set_ylim(0, 1.05)
ax_parse.legend(fontsize=7)
for bar in list(bars_ps) + list(bars_pf):
    h = bar.get_height()
    if h > 0.01:
        ax_parse.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                      f"{h:.2f}", ha="center", va="bottom", fontsize=7)

# Summary table
col_labels = ["Method", "EM strict", "EM flex", "Parse flex"]
rows = [[l, f"{em_strict[l]:.3f}", f"{em_flex[l]:.3f}", f"{pr_flex[l]:.3f}"]
        for l in labels]
tbl = ax_table.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)
ax_table.set_title("Summary (n=1319)", pad=12)

fig.suptitle("OPD softmax variants on GSM8K  —  Qwen2-0.5B student, Qwen2-1.5B teacher", fontsize=13)

out = os.path.join(BASE, "results_summary_softmax.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

print("\n── Results (n=1319) ──")
print(f"{'Method':<22} {'EM strict':>10} {'EM flex':>9} {'Parse strict':>13} {'Parse flex':>11}")
print("-" * 70)
for l in labels:
    print(f"{l:<22} {em_strict[l]:>10.3f} {em_flex[l]:>9.3f} {pr_strict[l]:>13.3f} {pr_flex[l]:>11.3f}")
