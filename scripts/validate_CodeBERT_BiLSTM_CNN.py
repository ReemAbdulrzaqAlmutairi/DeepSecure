#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_CodeBERT_BiLSTM_CNN.py — Validation / Calibration
----------------------------------------------------------
Validates the best checkpoint on the *validation* split ("test" key in the tokenized file),
computes ROC/PR metrics, finds the BEST threshold (max F1 over PR), evaluates at 0.5 and best
thresholds, measures simple efficiency (elapsed/throughput, peak VRAM), and writes:

  • results/best_threshold_{BASE}.txt
  • results/validation_results_{BASE}.txt
  • results/validation_results_{BASE}_thr050.txt
  • results/validation_results_{BASE}_thrbest.txt
  • artifacts/y_true_val_{BASE}.npy, artifacts/y_prob_val_{BASE}.npy
  • plots (ROC, PR, confusion matrices, calibration), unless SKIP_PLOTS=1

Design notes:
- Labels are assumed to be normalized in tokenize_CodeBERT.py (1=vulnerable, 0=clean).
- We do NOT flip labels here.
- ECE (Expected Calibration Error) is computed for reporting/calibration analysis.
"""

# ---- Quiet header & env flags ------------------------------------------------
import os, warnings, json
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TRANSFORMERS_NO_TF"]     = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = os.getenv("QUIET", "1")
warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
# -----------------------------------------------------------------------------

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

# ---- Config pulled from env (aligned with train script) ----------------------
BACKBONE     = os.getenv("BACKBONE", "microsoft/codebert-base")
VAL_WORKERS  = int(os.getenv("VAL_WORKERS", "0"))
PIN_MEMORY   = os.getenv("PIN_MEMORY", "1") == "1"
SKIP_PLOTS   = os.getenv("SKIP_PLOTS", "0") == "1"

# Import shared classes and some defaults from training script
from train_CodeBERT_BiLSTM_CNN import CodeDataset, CodeBERT_BiLSTM_CNN, BATCH_SIZE, MODE

TRAIN_DATASET = os.getenv("TRAIN_DATASET", "vudenc")
TEST_DATASET  = os.getenv("TEST_DATASET",  "vudenc")
RUNNAME       = os.getenv("RUNNAME",       "default")
BASE          = f"{MODE}_train-{TRAIN_DATASET}_test-{TEST_DATASET}_{RUNNAME}"

# ---- Metrics helpers ---------------------------------------------------------
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) over uniform probability bins.
    Returns a scalar in [0, 1], lower is better (closer to perfect calibration).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    # ECE (mean absolute gap between predicted prob and observed frequency)
    return float(np.mean(np.abs(prob_true - prob_pred)))

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """
    Run inference on the provided loader, return metrics at the given threshold and
    raw arrays for downstream analysis (y_true, y_prob).
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    # Measure total elapsed and peak VRAM (GPU only)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    start_ts = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_ts   = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if start_ts is not None:
        start_ts.record()

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        logits = model(input_ids, attention_mask)
        probs  = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        preds  = (probs >= threshold).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.long().cpu().numpy().tolist())

    if end_ts is not None:
        end_ts.record()
        torch.cuda.synchronize()
        total_ms = start_ts.elapsed_time(end_ts)
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        total_ms = 0.0
        peak_vram = 0.0

    # Scalar metrics at the provided threshold
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)

    # ROC-AUC (threshold-free)
    try:
        auc = roc_auc_score(all_labels, all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
    except ValueError:
        auc = 0.5
        fpr, tpr = np.array([0, 1]), np.array([0, 1])

    return (
        (acc, prec, rec, f1, auc),
        total_ms,
        peak_vram,
        (np.array(all_labels, dtype=int), np.array(all_probs, dtype=float)),
        (fpr, tpr)
    )

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | BACKBONE={BACKBONE} | BASE={BASE} | "
          f"BATCH_SIZE={BATCH_SIZE} | VAL_WORKERS={VAL_WORKERS} | PIN_MEMORY={PIN_MEMORY}")

    # Load tokenized splits
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
    print("Validation labels distribution:", Counter(data['test']['labels']))

    val_ds     = CodeDataset(data['test']['inputs'],  data['test']['labels'])
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=PIN_MEMORY, num_workers=VAL_WORKERS)

    # Load best checkpoint
    model = CodeBERT_BiLSTM_CNN(model_name=BACKBONE).to(device)
    ckpt_path = f'checkpoints/best_{BASE}.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} — train the model first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}")

    # Evaluate @ threshold=0.5
    (acc, prec, rec, f1, auc), inf_ms, peak_vram, (y_true, y_prob), (fpr, tpr) = evaluate(
        model, val_loader, device, threshold=0.5
    )
    ece_050 = compute_ece(y_true, y_prob, n_bins=10)

    # Basic throughput (samples/sec) over the *whole* validation pass
    num_samples = len(y_true)
    throughput  = num_samples / max(inf_ms / 1000.0, 1e-9)

    print(f"\nValidation for {BASE} @ thr=0.5:")
    print(f"  Accuracy  : {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f} | ECE: {ece_050:.4f}")
    print(f"  Time: {inf_ms/1000:.2f}s  | Throughput: {throughput:.2f} samples/s | Peak GPU RAM: {peak_vram:.1f} MiB")

    # Best threshold on validation via PR F1 maximization (for RQ3.1 Threshold Transfer)
    precs, recs, thrs = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precs * recs / (precs + recs + 1e-9)
    if len(thrs) > 0 and len(f1s) > 1:
        best_thr_idx = int(np.argmax(f1s[1:]))  # skip index 0 (no threshold)
        best_thr = float(thrs[best_thr_idx])
        best_val_f1_at_thr = float(np.max(f1s[1:]))
    else:
        best_thr = 0.5
        best_val_f1_at_thr = f1

    # Evaluate @ best_thr
    preds_best = (y_prob >= best_thr).astype(int)
    acc2  = accuracy_score(y_true, preds_best)
    prec2 = precision_score(y_true, preds_best, zero_division=0)
    rec2  = recall_score(y_true, preds_best, zero_division=0)
    f12   = f1_score(y_true, preds_best, zero_division=0)
    ece_best = compute_ece(y_true, y_prob, n_bins=10)  # ECE is threshold-free; kept for completeness

    print(f"\nValidation @ best thr={best_thr:.4f}: F1={f12:.4f}")

    # Save arrays, threshold, and text results
    Path('results').mkdir(exist_ok=True)
    Path('artifacts').mkdir(exist_ok=True)
    Path('plots').mkdir(exist_ok=True)

    np.save(f'artifacts/y_true_val_{BASE}.npy', y_true)
    np.save(f'artifacts/y_prob_val_{BASE}.npy',  y_prob)

    with open(f'results/best_threshold_{BASE}.txt', 'w') as fh:
        fh.write(str(best_thr))
    print(f"Saved best threshold → results/best_threshold_{BASE}.txt")

    # Plots (if enabled)
    if not SKIP_PLOTS:
        # ROC
        plt.figure(figsize=(8, 5))
        plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.4f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (Validation) — {BASE}')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(f'plots/roc_curve_{BASE}_validation.png', dpi=300, bbox_inches='tight')
        plt.close()

        # PR
        PrecisionRecallDisplay.from_predictions(y_true, y_prob)
        plt.title(f'Precision-Recall (Validation) — {BASE}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'plots/pr_curve_{BASE}_validation.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Confusion matrices at 0.5 and best threshold
        for thr, tag in [(0.5, '050'), (best_thr, 'best')]:
            preds_thr = (y_prob >= thr).astype(int)
            cm = confusion_matrix(y_true, preds_thr, labels=[0, 1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Clean(0)', 'Vuln(1)'])
            disp.plot(values_format='d', cmap='Blues')
            plt.title(f'Confusion Matrix (Validation) thr={thr:.3f} — {BASE}')
            plt.savefig(f'plots/confmat_{BASE}_validation_{tag}.png', dpi=300, bbox_inches='tight')
            plt.close()

        # Calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
            plt.figure(figsize=(5, 5))
            plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='gray')
            plt.plot(prob_pred, prob_true, marker='o')
            plt.xlabel('Predicted probability'); plt.ylabel('Observed frequency')
            plt.title(f'Calibration (Validation) — {BASE}')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'plots/calibration_{BASE}_validation.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[WARN] Calibration curve skipped: {e}")

    # Write textual summaries (machine-friendly key=value)
    with open(f'results/validation_results_{BASE}.txt', 'w') as out:
        out.write(f"Validation for {BASE}\n")
        out.write(f"base: {BASE}\n")
        out.write(f"mode: {MODE}\n")
        out.write(f"backbone: {BACKBONE}\n")
        out.write(f"seed: {os.getenv('SEED','')}\n")
        out.write(f"batch_size: {BATCH_SIZE}\n\n")

        out.write(f"@ thr=0.5\n")
        out.write(f"accuracy: {acc:.4f}\n")
        out.write(f"precision: {prec:.4f}\n")
        out.write(f"recall: {rec:.4f}\n")
        out.write(f"f1: {f1:.4f}\n")
        out.write(f"roc_auc: {auc:.4f}\n")
        out.write(f"ece: {ece_050:.4f}\n")
        out.write(f"inference_time_ms: {inf_ms:.4f}\n")
        out.write(f"throughput: {throughput:.4f}\n")
        out.write(f"gpu_vram: {peak_vram:.1f}\n\n")

        out.write(f"Best Threshold (PR-F1 max): {best_thr:.4f}\n")
        out.write(f"best_val_f1_at_threshold: {best_val_f1_at_thr:.4f}\n")
        out.write(f"@ thr={best_thr:.4f}\n")
        out.write(f"accuracy: {acc2:.4f}\n")
        out.write(f"precision: {prec2:.4f}\n")
        out.write(f"recall: {rec2:.4f}\n")
        out.write(f"f1: {f12:.4f}\n")
        out.write(f"roc_auc: {auc:.4f}\n")
        out.write(f"ece: {ece_best:.4f}\n")

    with open(f'results/validation_results_{BASE}_thr050.txt', 'w') as out:
        out.write(f"Validation for {BASE} @ thr=0.5000\n")
        out.write(f"accuracy: {acc:.4f}\n")
        out.write(f"precision: {prec:.4f}\n")
        out.write(f"recall: {rec:.4f}\n")
        out.write(f"f1: {f1:.4f}\n")
        out.write(f"roc_auc: {auc:.4f}\n")
        out.write(f"ece: {ece_050:.4f}\n")
        out.write(f"inference_time_ms: {inf_ms:.4f}\n")
        out.write(f"throughput: {throughput:.4f}\n")
        out.write(f"gpu_vram: {peak_vram:.1f}\n")

    with open(f'results/validation_results_{BASE}_thrbest.txt', 'w') as out:
        out.write(f"Validation for {BASE} @ thr={best_thr:.4f}\n")
        out.write(f"accuracy: {acc2:.4f}\n")
        out.write(f"precision: {prec2:.4f}\n")
        out.write(f"recall: {rec2:.4f}\n")
        out.write(f"f1: {f12:.4f}\n")
        out.write(f"roc_auc: {auc:.4f}\n")
        out.write(f"ece: {ece_best:.4f}\n")
        out.write(f"inference_time_ms: {inf_ms:.4f}\n")
        out.write(f"throughput: {throughput:.4f}\n")
        out.write(f"gpu_vram: {peak_vram:.1f}\n")

    print(f"\nResults saved under results/ (main + thr050 + thrbest). Arrays under artifacts/. Plots under plots/.")

if __name__ == "__main__":
    main()
