#!/usr/bin/env python3
"""Print a table of all evaluated models and their flex GSM8K score."""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

rows = []
for json_path in sorted(SCRIPT_DIR.rglob("eval_gsm8k.json")):
    try:
        data = json.loads(json_path.read_text())
        flex = data.get("exact_match_flex")
        if flex is None:
            continue
        rows.append((json_path.parent.name, flex))
    except Exception:
        continue

rows.sort(key=lambda r: r[1], reverse=True)

col_w = max(len(r[0]) for r in rows)
print(f"{'Model':<{col_w}}  flex_gsm8k")
print("-" * (col_w + 13))
for name, flex in rows:
    print(f"{name:<{col_w}}  {flex:.4f}")
