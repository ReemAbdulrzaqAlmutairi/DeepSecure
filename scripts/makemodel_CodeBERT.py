#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
makemodel_CodeBERT.py — Data Builder (UPDATED)
----------------------------------------------
This script constructs fixed-length code blocks (windows) from raw JSON data
for a specific vulnerability mode (e.g., SQLi, XSS, Command Injection, etc.).
It:
  • Reads `plain_{MODE}.json` (repository → commits → files → changes)
  • Applies several filtering rules
  • Extracts vulnerable code "blocks" using sliding windows
  • Randomly splits blocks into train / validation / final test sets
  • Saves data and manifest files using the unified BASE naming scheme

Important:
- Labels (0/1) are NOT altered here. They will be normalized later in
  `tokenize_CodeBERT.py` to ensure 1 = Vulnerable, 0 = Clean.
- BASE ensures consistent file naming across all scripts and experiments.
"""

import os
import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import myutils  # must provide findposition / findpositions / getblocks

# =========[ Environment configuration ]=========
MODE         = os.getenv("MODE", "xss")          # Example: sql, xss, command_injection, rce, ...
SEED         = int(os.getenv("SEED", "42"))      # For reproducible shuffling
STEP         = int(os.getenv("STEP", "5"))
FULL_LENGTH  = int(os.getenv("FULL_LENGTH", "200"))

TRAIN_DATASET = os.getenv("TRAIN_DATASET", "vudenc")
TEST_DATASET  = os.getenv("TEST_DATASET", "vudenc")
RUNNAME       = os.getenv("RUNNAME", "default")

# Unified base identifier for all file outputs
BASE = f"{MODE}_train-{TRAIN_DATASET}_test-{TEST_DATASET}_{RUNNAME}"

# Split ratios (can be overridden via environment variables)
TRAIN_PCT = float(os.getenv("TRAIN_PCT", "0.70"))
VAL_PCT   = float(os.getenv("VAL_PCT",   "0.15"))  # Validation ("test" in legacy code)
TEST_PCT  = 1.0 - TRAIN_PCT - VAL_PCT              # Final test portion

# Filtering restrictions (matching paper defaults)
restriction = [
    int(os.getenv("MAX_CODE_LEN", "20000")),  # 0: max code length per file
    int(os.getenv("MAX_BADPARTS_PER_CHANGE", "5")),  # 1: max bad parts per change
    int(os.getenv("MAX_BADPARTS_PER_FILE", "6")),    # 2: max bad parts per file
    int(os.getenv("MAX_FILES_PER_COMMIT", "10")),    # 3: max files per commit
]

# =========[ Directories setup ]=========
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
PLOTS_DIR   = Path("plots")  # optional for later analysis
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SRC_PATH = DATA_DIR / f"plain_{MODE}.json"

# =========[ Helper functions ]=========

def _validate_splits() -> None:
    """Ensure split ratios are valid and sum to ~1."""
    if not (0 < TRAIN_PCT < 1) or not (0 < VAL_PCT < 1):
        raise ValueError("TRAIN_PCT and VAL_PCT must be within (0,1).")
    if abs(TRAIN_PCT + VAL_PCT + TEST_PCT - 1.0) > 1e-9:
        raise ValueError("Splits do not sum to 1. Check TRAIN_PCT and VAL_PCT values.")

def _load_raw_json(path: Path) -> Dict[str, Any]:
    """Load the raw JSON file containing repository → commits → files → changes."""
    if not path.is_file():
        raise FileNotFoundError(f"Expected {path}. Please prepare plain_{MODE}.json first.")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"{path} is empty or malformed.")
    return data

def _extract_blocks(data: Dict[str, Any]) -> (List[Any], Dict[str, float]):
    """
    Iterate through repositories and commits to generate code blocks.

    Filtering steps:
      1. Skip commits with too many files.
      2. Skip files exceeding MAX_CODE_LEN.
      3. Skip changes exceeding MAX_BADPARTS limits.
      4. Use myutils.findpositions() to locate vulnerable snippets.
      5. Use myutils.getblocks() to build fixed-length sliding windows.

    Returns:
        allblocks (list): list of code windows (blocks)
        stats (dict): summary statistics
    """
    random.seed(SEED)
    allblocks = []
    repos_cnt = len(data)
    commits_cnt = 0
    files_used = 0
    blocks_per_file = []

    for repo, commits in data.items():
        for _commit_sha, finfo_by_file in commits.items():
            commits_cnt += 1
            files = finfo_by_file.get("files", {})
            if not files:
                continue

            # Skip commits with too many files
            if len(files) > restriction[3]:
                continue

            for fname, finfo in files.items():
                if "changes" not in finfo or "source" not in finfo:
                    continue

                sourcecode = finfo["source"]
                if not isinstance(sourcecode, str) or len(sourcecode) == 0:
                    continue
                if len(sourcecode) > restriction[0]:
                    continue

                # Collect vulnerable snippets (badparts) per file
                allbadparts = []
                skip_file = False

                for change in finfo["changes"]:
                    badparts = change.get("badparts", [])
                    if len(badparts) > restriction[1]:
                        skip_file = True
                        break
                    for bad in badparts:
                        pos = myutils.findposition(bad, sourcecode)
                        if -1 not in pos:
                            allbadparts.append(bad)
                    if len(allbadparts) > restriction[2]:
                        skip_file = True
                        break

                if skip_file or not allbadparts:
                    continue

                # Convert bad parts → positions → blocks
                positions = myutils.findpositions(allbadparts, sourcecode)
                blocks = myutils.getblocks(sourcecode, positions, STEP, FULL_LENGTH)
                if blocks:
                    allblocks.extend(blocks)
                    files_used += 1
                    blocks_per_file.append(len(blocks))

    stats = {
        "repos_count": repos_cnt,
        "commits_seen": commits_cnt,
        "files_used": files_used,
        "avg_blocks_per_file": (sum(blocks_per_file) / max(len(blocks_per_file), 1)),
    }
    return allblocks, stats

def _split_indices(n: int) -> Dict[str, List[int]]:
    """Randomly shuffle and split block indices into train / val / final sets."""
    keys = list(range(n))
    random.seed(SEED)
    random.shuffle(keys)

    train_cut = round(TRAIN_PCT * n)
    val_cut   = round((TRAIN_PCT + VAL_PCT) * n)

    return {
        "train": keys[:train_cut],
        "val": keys[train_cut:val_cut],
        "final": keys[val_cut:]
    }

# =========[ Main entry point ]=========

def main() -> None:
    # 1) Validate splits and load raw data
    _validate_splits()
    raw = _load_raw_json(SRC_PATH)
    print(f"[{MODE}] Loaded raw JSON → {SRC_PATH} (repos={len(raw)})")

    # 2) Extract blocks
    allblocks, stats = _extract_blocks(raw)
    n_blocks = len(allblocks)
    print(f"[{MODE}] Generated {n_blocks} blocks from {stats['files_used']} files "
          f"(commits≈{stats['commits_seen']})")

    if n_blocks == 0:
        raise RuntimeError("No blocks were generated. Check data quality or restrictions.")

    # 3) Split into train / val / final sets
    splits = _split_indices(n_blocks)
    print(f"[{MODE}] Split sizes → train={len(splits['train'])} | val={len(splits['val'])} | final={len(splits['final'])}")

    # 4) Save outputs (pickle + JSON)
    (DATA_DIR / f"{BASE}_blocks.pkl").write_bytes(pickle.dumps(allblocks))
    (DATA_DIR / f"{BASE}_dataset_keystrain").write_bytes(pickle.dumps(splits["train"]))
    (DATA_DIR / f"{BASE}_dataset_keystest").write_bytes(pickle.dumps(splits["val"]))
    (DATA_DIR / f"{BASE}_dataset_keysfinaltest").write_bytes(pickle.dumps(splits["final"]))

    with (DATA_DIR / f"{BASE}_splits.json").open("w", encoding="utf-8") as js:
        json.dump(splits, js, ensure_ascii=False, indent=2)

    # 5) Write manifest (metadata summary)
    manifest = {
        "mode": MODE,
        "base": BASE,
        "seed": SEED,
        "step": STEP,
        "full_length": FULL_LENGTH,
        "restriction": {
            "max_code_len": restriction[0],
            "max_badparts_per_change": restriction[1],
            "max_badparts_per_file": restriction[2],
            "max_files_per_commit": restriction[3],
        },
        "splits": {
            "train_pct": TRAIN_PCT,
            "val_pct": VAL_PCT,
            "final_pct": TEST_PCT,
            "train_size": len(splits["train"]),
            "val_size": len(splits["val"]),
            "final_size": len(splits["final"]),
            "total_blocks": n_blocks,
        },
        "data_stats": stats,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source_file": str(SRC_PATH),
        "notes": (
            "Labels are NOT altered here. "
            "Normalization (1=vulnerable, 0=clean) occurs later in tokenize_CodeBERT.py."
        ),
    }

    with (RESULTS_DIR / f"manifest_data_{BASE}.json").open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    # 6) Friendly summary printout
    print("-" * 60)
    print(f"✅ Done building windows for [{MODE}]")
    print(f"• BASE: {BASE}")
    print(f"• Blocks saved       → data/{BASE}_blocks.pkl")
    print(f"• Splits (pickle)    → data/{BASE}_dataset_keystrain | _keystest | _keysfinaltest")
    print(f"• Splits (JSON)      → data/{BASE}_splits.json")
    print(f"• Manifest           → results/manifest_data_{BASE}.json")
    print("-" * 60)

if __name__ == "__main__":
    main()
