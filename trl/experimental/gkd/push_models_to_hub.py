#!/usr/bin/env python3
"""
Push the 13 evaluated models to HuggingFace organisation QpiEImitation.

Auto-discovers the most recent evaluated directory (must contain eval_gsm8k.json)
for each of the 13 runs:
  - 1  GKD baseline
  - 4  OPD-Entropy    (no_corr/no_tr | no_corr/tr | corr/no_tr | corr/tr)
  - 4  OPD-Expectation
  - 4  OPD-Stochastic

Repo name = QpiEImitation/<folder-name-without-timestamp>

Files excluded from upload: *.log, *.png, *.jsonl, training_args.bin
"""

import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, login

BASE = Path(__file__).parent.resolve()
ORG  = "QpiEImitation"

IGNORE_PATTERNS = ["*.log", "*.png", "*.jsonl", "training_args.bin"]

# Timestamp suffix pattern: _YYYYMMDD_HHMMSS
_TS_RE = re.compile(r"_\d{8}_\d{6}$")


def find_latest(pattern: str) -> Path | None:
    candidates = sorted(BASE.glob(pattern), key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def repo_name(folder: Path) -> str:
    return _TS_RE.sub("", folder.name)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

BASE_PAT = "opd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct"
GKD_PAT  = "gkd_gsm8k_S-Qwen2-0.5B-Instruct_T-Qwen2-1.5B-Instruct"

PATTERNS = [
    ("GKD",             False, False, f"{GKD_PAT}_*"),
    ("OPD-Entropy",     False, False, f"{BASE_PAT}_entropy_baseline_L*_buf*"),
    ("OPD-Entropy",     True,  False, f"{BASE_PAT}_entropy_baseline_tr_L*_buf*"),
    ("OPD-Entropy",     False, True,  f"{BASE_PAT}_entropy_baseline_corr_L*_buf*"),
    ("OPD-Entropy",     True,  True,  f"{BASE_PAT}_entropy_baseline_tr_corr_L*_buf*"),
    ("OPD-Expectation", False, False, f"{BASE_PAT}_expectation_L*_buf*"),
    ("OPD-Expectation", True,  False, f"{BASE_PAT}_expectation_tr_L*_buf*"),
    ("OPD-Expectation", False, True,  f"{BASE_PAT}_expectation_corr_L*_buf*"),
    ("OPD-Expectation", True,  True,  f"{BASE_PAT}_expectation_tr_corr_L*_buf*"),
    ("OPD-Stochastic",  False, False, f"{BASE_PAT}_stochastic_L*_buf*"),
    ("OPD-Stochastic",  True,  False, f"{BASE_PAT}_stochastic_tr_L*_buf*"),
    ("OPD-Stochastic",  False, True,  f"{BASE_PAT}_stochastic_corr_L*_buf*"),
    ("OPD-Stochastic",  True,  True,  f"{BASE_PAT}_stochastic_tr_corr_L*_buf*"),
]

jobs: list[tuple[str, Path]] = []
missing: list[str] = []

print("Discovering evaluated model directories:")
for algo, tr, corr, pattern in PATTERNS:
    folder = find_latest(pattern)
    label  = f"{algo:20s}  TR={int(tr)}  Corr={int(corr)}"
    if folder is None:
        print(f"  [MISSING]  {label}  →  no dir for pattern: {pattern}")
        missing.append(label)
    else:
        name = repo_name(folder)
        print(f"  [OK]       {label}  →  {folder.name}")
        jobs.append((name, folder))

if missing:
    print(f"\n[WARNING] {len(missing)} model(s) not found and will be skipped.")

print(f"\n{len(jobs)} models will be pushed to organisation '{ORG}'.")
print("Repos that will be created / updated:")
for name, folder in jobs:
    print(f"  {ORG}/{name}")

confirm = input("\nProceed? [y/N] ").strip().lower()
if confirm != "y":
    print("Aborted.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Login & push
# ---------------------------------------------------------------------------

token_path = Path.home() / ".cache/huggingface/token"
token = token_path.read_text().strip() if token_path.exists() else None
login(token=token)

api = HfApi()

for i, (name, folder) in enumerate(jobs, 1):
    repo_id = f"{ORG}/{name}"
    print(f"\n[{i}/{len(jobs)}] Pushing {folder.name}  →  {repo_id}")

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=f"Upload {name}",
    )
    print(f"  Done → https://huggingface.co/{repo_id}")

print(f"\nAll {len(jobs)} models pushed successfully.")
