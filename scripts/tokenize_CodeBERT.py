#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tokenize_CodeBERT.py — Tokenizer & PKL Builder (UPDATED)
--------------------------------------------------------
What this script does:
  • Loads blocks & split indices produced by makemodel_CodeBERT.py (via BASE names)
  • Tokenizes code using a Hugging Face tokenizer (CodeBERT/GraphCodeBERT)
  • Normalizes labels ONCE here: 1 = vulnerable, 0 = clean
  • Saves a compact PKL for train / validation ("test") / final test
  • Emits a manifest with useful stats for reproducibility and reporting

Design notes:
- We intentionally normalize labels here so all downstream scripts do NOT flip labels.
- MAX_LEN kept aligned with your paper’s window length (200 by default).
"""

# ---- Quiet header & env knobs ------------------------------------------------
import os, json, pickle, warnings, hashlib
from collections import Counter
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = os.getenv("QUIET", "0")  # toggle tqdm
warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
# -----------------------------------------------------------------------------


from pathlib import Path
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer
from tqdm import tqdm


# =========[ Environment configuration ]=========
MODE         = os.getenv("MODE", "xss")
BACKBONE_RAW = os.getenv("BACKBONE", "microsoft/codebert-base")
BACKBONE     = BACKBONE_RAW.split("#")[0].strip()  # allow "#rev" suffix in env if needed
MAX_LEN      = int(os.getenv("MAX_SEQ_LEN", "200"))   # keep in sync with paper (200)
TOK_BATCH    = int(os.getenv("TOK_BATCH", "1024"))
CACHE_DIR    = os.getenv("HF_CACHE_DIR", "").strip() or None
QUIET        = os.getenv("QUIET", "0") == "1"

TRAIN_DATASET = os.getenv("TRAIN_DATASET", "vudenc")
TEST_DATASET  = os.getenv("TEST_DATASET",  "vudenc")
RUNNAME       = os.getenv("RUNNAME",       "default")

# Unified BASE for all inputs/outputs
BASE = f"{MODE}_train-{TRAIN_DATASET}_test-{TEST_DATASET}_{RUNNAME}"

# =========[ Inputs produced by makemodel (BASE) ]=========
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

blocks_path     = DATA_DIR / f"{BASE}_blocks.pkl"
keys_train_path = DATA_DIR / f"{BASE}_dataset_keystrain"
keys_val_path   = DATA_DIR / f"{BASE}_dataset_keystest"       # our "validation" split
keys_final_path = DATA_DIR / f"{BASE}_dataset_keysfinaltest"  # final test split

for p in [blocks_path, keys_train_path, keys_val_path, keys_final_path]:
    if not p.is_file():
        raise FileNotFoundError(f"Missing file: {p}  — run makemodel_CodeBERT.py first (BASE={BASE}).")

# Load blocks & indices
blocks = pickle.loads(blocks_path.read_bytes())
keystrain      = pickle.loads(keys_train_path.read_bytes())
keystest       = pickle.loads(keys_val_path.read_bytes())
keysfinaltest  = pickle.loads(keys_final_path.read_bytes())

print(f"[{BASE}] blocks={len(blocks)} | train={len(keystrain)} | val={len(keystest)} | final={len(keysfinaltest)}")
print(f"Tokenizer: {BACKBONE} | MAX_LEN={MAX_LEN} | TOK_BATCH={TOK_BATCH}")


# =========[ Load tokenizer ]=========
tok_kwargs: Dict[str, Any] = {"use_fast": True}
if CACHE_DIR:
    tok_kwargs["cache_dir"] = CACHE_DIR
tokenizer = AutoTokenizer.from_pretrained(BACKBONE, **tok_kwargs)


# =========[ Label normalization policy ]=========
# Source dataset (legacy) had 0=vulnerable, 1=clean (per original codebase).
# We normalize ONCE here to enforce 1=vulnerable, 0=clean for ALL downstream scripts.
LABEL_ORIGIN_POLICY = os.getenv("LABEL_ORIGIN_POLICY", "legacy_0_vuln")  # keep default legacy for VUDENC/Bagheri

def normalize_label(l: int) -> int:
    """
    Map label to (1=vulnerable, 0=clean).

    If LABEL_ORIGIN_POLICY == "legacy_0_vuln" (default for VUDENC/Bagheri), flip 0<->1.
    If LABEL_ORIGIN_POLICY == "native_1_vuln", keep as-is.
    """
    l = int(l)
    if l not in (0, 1):
        raise ValueError(f"Non-binary label detected: {l}")

    if LABEL_ORIGIN_POLICY == "legacy_0_vuln":
        return 1 - l
    elif LABEL_ORIGIN_POLICY == "native_1_vuln":
        return l
    else:
        raise ValueError(f"Unknown LABEL_ORIGIN_POLICY: {LABEL_ORIGIN_POLICY}")


def _extract_text_and_labels(keys: List[int]) -> Tuple[List[str], List[int], List[int]]:
    """Collect raw texts, normalized labels, and original labels for a given split."""
    texts, labels_norm, labels_orig = [], [], []
    for k in keys:
        code, label_orig = blocks[k]  # (str, int)
        texts.append(code)
        labels_orig.append(int(label_orig))
        labels_norm.append(normalize_label(label_orig))
    return texts, labels_norm, labels_orig


def _encode_bulk(texts: List[str]) -> Dict[str, List[List[int]]]:
    """Tokenize a list of code snippets with fixed-length truncation/padding."""
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_attention_mask=True,
    )


def _pack_split(keys: List[int], desc: str) -> Tuple[List[Dict[str, List[int]]], List[int], Dict[str, Any], List[int]]:
    """
    Encode a split and compute basic stats.
    Returns:
        inputs: list of dicts {"input_ids": [...], "attention_mask": [...]}
        labels_norm: list[int] normalized to (1=vulnerable, 0=clean)
        stats: dict with size, avg effective length, truncation ratio, label distribution
        labels_orig: original labels (before normalization) for checksum/reporting
    """
    texts, labels_norm, labels_orig = _extract_text_and_labels(keys)

    # Sanity check after normalization
    bad = [x for x in labels_norm if x not in (0, 1)]
    if bad:
        raise AssertionError(f"{desc}: found non-binary labels after normalization: {set(bad)}")

    inputs, attn_sums, maxlen_hits = [], [], 0
    rng = range(0, len(texts), TOK_BATCH)
    iterator = rng if QUIET else tqdm(rng, desc=f"Tokenizing [{desc}]", ncols=80)
    for i in iterator:
        batch_texts = texts[i:i + TOK_BATCH]
        enc = _encode_bulk(batch_texts)
        for j in range(len(batch_texts)):
            input_ids = enc["input_ids"][j]
            attn_mask = enc["attention_mask"][j]
            inputs.append({"input_ids": input_ids, "attention_mask": attn_mask})

            eff_len = int(sum(attn_mask))  # tokens before padding
            attn_sums.append(eff_len)
            if eff_len >= MAX_LEN:
                maxlen_hits += 1

    stats = {
        "size": len(keys),
        "avg_effective_len": round(float(sum(attn_sums)) / max(1, len(attn_sums)), 3),
        "maxlen_ratio": round(maxlen_hits / max(1, len(attn_sums)), 4),
        "label_dist": dict(Counter(labels_norm)),
    }
    return inputs, labels_norm, stats, labels_orig


# =========[ Encode splits ]=========
train_inputs, train_labels, train_stats, train_labels_orig   = _pack_split(keystrain, "train")
val_inputs,   val_labels,   val_stats,   val_labels_orig     = _pack_split(keystest,  "val")
final_inputs, final_labels, final_stats, final_labels_orig   = _pack_split(keysfinaltest, "final")

# =========[ Save tokenized object ]=========
tokenized_obj = {
    "train":     {"inputs": train_inputs,  "labels": train_labels},
    "test":      {"inputs": val_inputs,    "labels": val_labels},      # "test" == validation split (legacy naming)
    "finaltest": {"inputs": final_inputs,  "labels": final_labels},
}
out_pkl = DATA_DIR / f"{BASE}_tokenized.pkl"
out_pkl.write_bytes(pickle.dumps(tokenized_obj))
print("Saved →", out_pkl)


# =========[ Write manifest (two locations for convenience) ]=========
# Build a checksum proving one-time normalization from original->normalized across all splits
all_orig  = train_labels_orig + val_labels_orig + final_labels_orig
all_norm  = train_labels      + val_labels      + final_labels
checksum  = hashlib.sha256(("".join(map(str, all_orig)) + "->" + "".join(map(str, all_norm))).encode()).hexdigest()

manifest = {
    "mode": MODE,
    "base": BASE,
    "backbone": BACKBONE,
    "max_seq_len": MAX_LEN,
    "tok_batch": TOK_BATCH,

    "train_stats": train_stats,
    "val_stats":   val_stats,
    "final_stats": final_stats,

    # Label policy documentation
    "label_origin_policy": LABEL_ORIGIN_POLICY,                  # "legacy_0_vuln" (default) or "native_1_vuln"
    "label_policy_applied": "normalized_to_1=vuln_0=clean",      # enforced in this script
    "normalized_at": "tokenize",
    "normalized_once_checksum": checksum
}

# (A) A manifest colocated with tokenized.pkl for downstream scripts to assert quickly
manifest_data_path = DATA_DIR / f"{BASE}_tokenized_manifest.json"
manifest_data_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("Manifest (data) →", manifest_data_path)

# (B) A copy in results/ for easy human browsing
manifest_results_path = RESULTS_DIR / f"manifest_tokenizer_{BASE}.json"
manifest_results_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("Manifest (results) →", manifest_results_path)


# =========[ Friendly warnings / tips ]=========
max_trunc = max(train_stats["maxlen_ratio"], val_stats["maxlen_ratio"], final_stats["maxlen_ratio"])
if max_trunc > 0.25:
    print(f"[WARN] High truncation ratio (>25%). Consider raising MAX_SEQ_LEN (current {MAX_LEN}).")

def _fmt_dist(d: Dict[int, int]) -> str:
    total = sum(d.values()) or 1
    return ", ".join([f"{k}:{v} ({v/total:.1%})" for k, v in sorted(d.items())])

print(f"Label dist — train: {_fmt_dist(train_stats['label_dist'])} | "
      f"val: {_fmt_dist(val_stats['label_dist'])} | final: {_fmt_dist(final_stats['label_dist'])}")

print("✅ Done Tokenizing")
