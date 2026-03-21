#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test-CodeBERT_BiLSTM_CNN.py — Final Test Evaluation (UPDATED)
--------------------------------------------------------------
Evaluates the best checkpoint on the FINAL test split ("finaltest" in the tokenized file).
Outputs:
  • results/final_test_results_{BASE}.txt      (human-readable, parsable key=val)
  • results/final_test_results_{BASE}_thr050.txt
  • results/final_test_results_{BASE}_thrbest.txt
  • results/final_test_results_{BASE}.json     (machine-readable)
  • artifacts/y_true_final_{BASE}.npy
  • artifacts/y_prob_final_{BASE}.npy
  • plots/ (PR, ROC, confusion matrices, calibration) unless SKIP_PLOTS=1

Notes:
- Labels MUST already be normalized in tokenize step (1=vulnerable, 0=clean).
- No label flipping here.
- Best validation threshold is read from results/best_threshold_{BASE}.txt if present.
"""

# ===== Quiet header (keep at VERY TOP) =====
import os, warnings, logging, json
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_TF"]    = "1"
os.environ["TOKENIZERS_PARALLELISM"]= "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = os.getenv("QUIET", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.getLogger("absl").setLevel(logging.ERROR)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
# ===========================================

from datetime import datetime, timezone
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay
)
from sklearn.calibration import calibration_curve

# Env flags
BACKBONE    = os.getenv("BACKBONE", "microsoft/codebert-base")
VAL_WORKERS = int(os.getenv("VAL_WORKERS", "0"))
PIN_MEMORY  = os.getenv("PIN_MEMORY", "1") == "1"
SKIP_PLOTS  = os.getenv("SKIP_PLOTS", "0") == "1"

# Reuse dataset/model/batch from the training script
from train_CodeBERT_BiLSTM_CNN import CodeDataset, CodeBERT_BiLSTM_CNN, BATCH_SIZE, MODE

TRAIN_DATASET = os.getenv("TRAIN_DATASET", "vudenc")
TEST_DATASET  = os.getenv("TEST_DATASET",  "vudenc")
RUNNAME       = os.getenv("RUNNAME",       "default")
BASE          = f"{MODE}_train-{TRAIN_DATASET}_test-{TEST_DATASET}_{RUNNAME}"

# ---------- Helpers ----------
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (uniform bins).
    Lower is better. Threshold-free metric (based on probabilities).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    return float(np.mean(np.abs(prob_true - prob_pred)))

@torch.no_grad()
def infer_probs(model, loader, device):
    """
    Run forward pass on the entire loader.
    Returns: y_true, y_prob, total_inference_time_ms, peak_vram_mb
    """
    model.eval()
    all_labels, all_probs = [], []

    # GPU timing + peak VRAM (if available)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
    else:
        starter = ender = None

    for batch in loader:
        ids  = batch['input_ids'].to(device)
        am   = batch['attention_mask'].to(device)
        labs = batch['label'].long().cpu().numpy()

        logits = model(ids, am)
        probs  = torch.sigmoid(logits).cpu().numpy().reshape(-1)

        all_labels.extend(labs.tolist())
        all_probs.extend(probs.tolist())

    if starter is not None:
        ender.record()
        torch.cuda.synchronize()
        inf_time_ms = starter.elapsed_time(ender)
        peak_vram   = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        inf_time_ms = 0.0
        peak_vram   = 0.0

    return np.array(all_labels, dtype=int), np.array(all_probs, dtype=float), float(inf_time_ms), float(peak_vram)

def metrics_at_threshold(y_true, y_prob, thr):
    """Return (acc, prec, rec, f1) at a given threshold using >= (consistent)."""
    preds = (y_prob >= thr).astype(int)
    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    f1   = f1_score(y_true, preds, zero_division=0)
    return acc, prec, rec, f1

# ---------- Main ----------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | BACKBONE={BACKBONE} | BASE={BASE} | BATCH_SIZE={BATCH_SIZE} | VAL_WORKERS={VAL_WORKERS}")

    # Load tokenized data
    tokenized_path = f'data/{BASE}_tokenized.pkl'
    if not os.path.exists(tokenized_path):
        raise FileNotFoundError(f"Tokenized data not found: {tokenized_path} — run tokenize_CodeBERT.py first.")

    # ✅ Safety check: ensure labels were normalized ONCE in tokenize step
    manifest_path = f"data/{BASE}_tokenized_manifest.json"
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                m = json.load(mf)
            assert m.get("normalized_at") == "tokenize", \
                f"[ERROR] Labels must be normalized in tokenize step (found: {m.get('normalized_at')})"
            print(f"[CHECK] Label normalization verified via manifest ({manifest_path})")
        except Exception as e:
            print(f"[WARN] Could not verify manifest normalization: {e}")
    else:
        print(f"[WARN] Manifest not found for BASE={BASE}. Proceeding without verification.")

    with open(tokenized_path, 'rb') as f:
        data = pickle.load(f)

    # Labels were normalized in tokenize step → do NOT flip here
    print("FinalTest labels distribution:", Counter(data['finaltest']['labels']))

    final_ds     = CodeDataset(data['finaltest']['inputs'],  data['finaltest']['labels'])
    final_loader = DataLoader(final_ds, batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=PIN_MEMORY, num_workers=VAL_WORKERS)

    # Load model
    model = CodeBERT_BiLSTM_CNN(model_name=BACKBONE).to(device)
    ckpt_path = f'checkpoints/best_{BASE}.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} — train the model first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}")

    # Validation-selected threshold (if available)
    thr_path = f'results/best_threshold_{BASE}.txt'
    if os.path.isfile(thr_path):
        try:
            thr_opt = float(open(thr_path, 'r').read().strip())
        except Exception:
            thr_opt = 0.5
    else:
        print(f"[WARN] best_threshold file not found for {BASE}, using 0.5")
        thr_opt = 0.5

    # Inference on FINAL split
    y_true, y_prob, inf_ms, peak_vram = infer_probs(model, final_loader, device)
    throughput = len(y_true) / max(inf_ms/1000.0, 1e-9)

    # ROC-AUC + ROC curve
    try:
        auc = float(roc_auc_score(y_true, y_prob))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except Exception:
        auc = 0.5
        fpr, tpr = np.array([0, 1]), np.array([0, 1])

    # ECE (threshold-free)
    ece = compute_ece(y_true, y_prob, n_bins=10)

    # Metrics at 0.5 and at best validation threshold
    thr_05  = 0.5
    acc05, p05, r05, f105 = metrics_at_threshold(y_true, y_prob, thr_05)
    accop, pop, rop, f1op = metrics_at_threshold(y_true, y_prob, thr_opt)

    print(f"\nFinal Test for {BASE}:")
    print(f"  @ thr=0.5000 -> F1={f105:.4f}")
    print(f"  @ thr={thr_opt:.4f} (from validation) -> F1={f1op:.4f}")
    print(f"  ROC-AUC: {auc:.4f} | ECE: {ece:.4f}")
    print(f"  Inference: {inf_ms/1000:.2f}s total | Throughput: {throughput:.2f} samples/s | Peak GPU RAM: {peak_vram:.1f} MiB")

    # Ensure dirs
    Path('plots').mkdir(exist_ok=True)
    Path('artifacts').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    # Save arrays
    np.save(f'artifacts/y_true_final_{BASE}.npy', y_true)
    np.save(f'artifacts/y_prob_final_{BASE}.npy',  y_prob)

    # Plots (optional)
    if not SKIP_PLOTS:
        try:
            PrecisionRecallDisplay.from_predictions(y_true, y_prob)
            plt.title(f'Precision-Recall (Final Test) — {BASE}')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'plots/pr_curve_finaltest_{BASE}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[WARN] PR curve skipped: {e}")

        try:
            plt.figure(figsize=(8, 5))
            plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve (Final Test) — {BASE}')
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig(f'plots/roc_curve_finaltest_{BASE}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[WARN] ROC curve skipped: {e}")

        try:
            for thr, tag in [(thr_05, "thr050"), (thr_opt, "thrbest")]:
                preds = (y_prob >= thr).astype(int)
                cm = confusion_matrix(y_true, preds, labels=[0, 1])
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Clean(0)','Vuln(1)'])
                disp.plot(values_format='d', cmap='Blues')
                plt.title(f'Confusion Matrix (Final) {tag} — {BASE}')
                plt.savefig(f'plots/confmat_finaltest_{BASE}_{tag}.png', dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"[WARN] Confusion matrix skipped: {e}")

        try:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
            plt.figure(figsize=(5, 5))
            plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='gray')
            plt.plot(prob_pred, prob_true, marker='o')
            plt.xlabel('Predicted probability'); plt.ylabel('Observed frequency')
            plt.title(f'Calibration (Final Test) — {BASE}')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'plots/calibration_finaltest_{BASE}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[WARN] Calibration curve skipped: {e}")

    # Textual results (parsable key=value)
    with open(f'results/final_test_results_{BASE}_thr050.txt', 'w') as out:
        out.write(f"Final Test for {BASE} @ thr=0.5000\n")
        out.write(f"base: {BASE}\n")
        out.write(f"mode: {MODE}\n")
        out.write(f"backbone: {BACKBONE}\n")
        out.write(f"seed: {os.getenv('SEED','')}\n")
        out.write(f"batch_size: {BATCH_SIZE}\n")
        out.write(f"accuracy: {acc05:.4f}\n")
        out.write(f"precision: {p05:.4f}\n")
        out.write(f"recall: {r05:.4f}\n")
        out.write(f"f1: {f105:.4f}\n")
        out.write(f"roc_auc: {auc:.4f}\n")
        out.write(f"ece: {ece:.4f}\n")
        out.write(f"inference_time_ms: {inf_ms:.4f}\n")
        out.write(f"throughput: {throughput:.4f}\n")
        out.write(f"gpu_vram: {peak_vram:.1f}\n")

    with open(f'results/final_test_results_{BASE}_thrbest.txt', 'w') as out:
        out.write(f"Final Test for {BASE} @ thr={thr_opt:.4f} (from validation)\n")
        out.write(f"base: {BASE}\n")
        out.write(f"mode: {MODE}\n")
        out.write(f"backbone: {BACKBONE}\n")
        out.write(f"seed: {os.getenv('SEED','')}\n")
        out.write(f"accuracy: {accop:.4f}\n")
        out.write(f"precision: {pop:.4f}\n")
        out.write(f"recall: {rop:.4f}\n")
        out.write(f"f1: {f1op:.4f}\n")
        out.write(f"roc_auc: {auc:.4f}\n")
        out.write(f"ece: {ece:.4f}\n")
        out.write(f"inference_time_ms: {inf_ms:.4f}\n")
        out.write(f"throughput: {throughput:.4f}\n")
        out.write(f"gpu_vram: {peak_vram:.1f}\n")

    with open(f'results/final_test_results_{BASE}.txt', 'w') as out:
        out.write(f"Final Test for {BASE}\n")
        out.write(f"base: {BASE}\n")
        out.write(f"mode: {MODE}\n")
        out.write(f"backbone: {BACKBONE}\n")
        out.write(f"seed: {os.getenv('SEED','')}\n")
        out.write(f"roc_auc: {auc:.4f}\n")
        out.write(f"ece: {ece:.4f}\n")
        out.write(f"inference_time_ms: {inf_ms:.4f}\n")
        out.write(f"throughput: {throughput:.4f}\n")
        out.write(f"gpu_vram: {peak_vram:.1f}\n\n")
        out.write(f"@ thr=0.5000 -> Acc={acc05:.4f}, Prec={p05:.4f}, Rec={r05:.4f}, F1={f105:.4f}\n")
        out.write(f"@ thr={thr_opt:.4f} (from val) -> Acc={accop:.4f}, Prec={pop:.4f}, Rec={rop:.4f}, F1={f1op:.4f}\n")

    # JSON summary for aggregation tools
    test_json = {
        "mode": MODE,
        "base": BASE,
        "backbone": BACKBONE,
        "seed": os.getenv("SEED", ""),
        "f1_thr050": float(f105),
        "f1_thr_best": float(f1op),
        "roc_auc": float(auc),
        "ece": float(ece),
        "inference_time_ms": float(inf_ms),
        "throughput": float(throughput),
        "peak_vram_mb": float(peak_vram),
        "best_threshold": float(thr_opt),
        "n_samples": int(len(y_true)),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    try:
        with open(f"results/final_test_results_{BASE}.json", "w", encoding="utf-8") as jf:
            json.dump(test_json, jf, indent=2)
        print(f"Wrote results/final_test_results_{BASE}.json")
    except Exception as e:
        print(f"[WARN] Could not write final JSON summary: {e}")

    print(f"\nResults saved to results/final_test_results_{BASE}.txt "
          f"(and *_thr050.txt, *_thrbest.txt, .json). Arrays under artifacts/, plots under plots/.")

if __name__ == "__main__":
    main()
