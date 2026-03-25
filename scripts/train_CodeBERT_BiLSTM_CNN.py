#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_GraphCodeBERT_BiLSTM_CNN.py
----------------------------
Training script for the hybrid model (GraphCodeBERT + BiLSTM + CNN).

Key design notes:
- Labels are assumed to be normalized already in tokenize_CodeBERT.py
  (1 = vulnerable, 0 = clean). No flipping is performed here.
- Validation monitoring uses threshold=0.5 for early stopping (stable & deterministic).
- We still compute the BEST validation threshold from the PR curve and persist it
  to 'results/best_threshold_{BASE}.txt', and log 'best_val_f1_at_threshold' into
  'results/training_summary_{BASE}.json' for RQ3.1 (threshold transfer).
- Optional inference measurements (latency/throughput/peak VRAM) are supported.

Environment knobs:
  MODE, BACKBONE, BATCH_SIZE, EPOCHS, LR, SEED, EARLY_STOP, TRAIN_WORKERS, VAL_WORKERS, PIN_MEMORY,
  AMP, QUIET, LOG_EVERY, TRAIN_DATASET, TEST_DATASET, RUNNAME, POOL_MODE, LSTM_HIDDEN, CNN_FILTERS,
  CNN_KERNELS, DROPOUT, RUN_TEST_AFTER, MEASURE_INFERENCE
"""

import os, warnings, logging, time, random, json, pickle
from datetime import datetime, timezone 
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, precision_recall_curve
)

# Reduce noisy logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("absl").setLevel(logging.ERROR)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# -------------------------
# Config (from env / defaults)
# -------------------------
MODE          = os.getenv("MODE", "sql")
BACKBONE      = os.getenv("BACKBONE", "microsoft/graphcodebert-base")
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "128"))
EPOCHS        = int(os.getenv("EPOCHS", "100"))
LR            = float(os.getenv("LR", "1e-5")) # Note: published experiments used LR=1e-5
SEED          = int(os.getenv("SEED", "42"))
EARLY_STOP    = int(os.getenv("EARLY_STOP", "5"))
TRAIN_WORKERS = int(os.getenv("TRAIN_WORKERS", "0"))
VAL_WORKERS   = int(os.getenv("VAL_WORKERS", "0"))
PIN_MEMORY    = os.getenv("PIN_MEMORY", "1") == "1"
SKIP_PLOTS    = os.getenv("SKIP_PLOTS", "0") == "1"
AMP_ENABLED   = (os.getenv("AMP", "1") == "1") and torch.cuda.is_available()
QUIET         = os.getenv("QUIET", "1") == "1"
LOG_EVERY     = int(os.getenv("LOG_EVERY", "1"))

TRAIN_DATASET = os.getenv("TRAIN_DATASET", "vudenc")
TEST_DATASET  = os.getenv("TEST_DATASET", "vudenc")
RUNNAME       = os.getenv("RUNNAME", "default")
BASE          = f"{MODE}_train-{TRAIN_DATASET}_test-{TEST_DATASET}_{RUNNAME}"

# Model hyperparams via env
POOL_MODE   = os.getenv("POOL_MODE", "token_bilstm")   # token_bilstm | cls | mean
LSTM_HIDDEN = int(os.getenv("LSTM_HIDDEN", "100"))
CNN_FILTERS = int(os.getenv("CNN_FILTERS", "64"))
CNN_KERNELS = os.getenv("CNN_KERNELS", "3,4,5")        # e.g. "3,4,5" or "7,8,9"
DROPOUT     = float(os.getenv("DROPOUT", "0.2"))

# Optional flags
RUN_TEST_AFTER     = os.getenv("RUN_TEST_AFTER", "0") == "1"
MEASURE_INFERENCE  = os.getenv("MEASURE_INFERENCE", "1") == "1"

def log(msg: str):
    if not QUIET:
        print(msg)

# -------------------------
# Seed + utils
# -------------------------
def set_seed(sd=SEED):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class CodeDataset(Dataset):
    """Thin dataset wrapper for already-tokenized inputs and 0/1 labels."""
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.inputs[idx]
        return {
            'input_ids':      torch.tensor(item['input_ids'],      dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'label':          torch.tensor(self.labels[idx],       dtype=torch.float)
        }

# -------------------------
# Model (flexible pooling head)
# -------------------------
class CodeBERT_BiLSTM_CNN(nn.Module):
    """
    Backbone: BERT encoder (e.g., CodeBERT/GraphCodeBERT)
    Head:
      - cls pooling: linear over [CLS]
      - mean pooling: linear over attention-masked mean
      - token_bilstm: BiLSTM over tokens + multiple Conv1D + max pool + linear
    """
    def __init__(self, model_name=BACKBONE, lstm_hidden=LSTM_HIDDEN,
                 cnn_channels=CNN_FILTERS, cnn_kernels=CNN_KERNELS, dropout_p=DROPOUT,
                 pool_mode=POOL_MODE):
        super().__init__()

        # Parse kernel list if passed as comma-separated string
        if isinstance(cnn_kernels, str):
            cnn_kernels = [int(x) for x in cnn_kernels.split(",") if x.strip()]

        self.pool_mode = pool_mode
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size

        if self.pool_mode not in ('cls', 'mean'):
            self.lstm = nn.LSTM(input_size=bert_hidden,
                                hidden_size=lstm_hidden,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)
            self.convs = nn.ModuleList([
                nn.Conv1d(in_channels=lstm_hidden*2, out_channels=cnn_channels, kernel_size=k, padding=k//2)
                for k in cnn_kernels
            ])
            self.final_dim = cnn_channels * len(cnn_kernels)
        else:
            self.final_dim = bert_hidden

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(dropout_p)
        self.fc   = nn.Linear(self.final_dim, 1)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B, T, H)

        if self.pool_mode == 'cls':
            cls_vec = bert_out[:, 0, :]                     # (B, H)
            x = self.drop(cls_vec)
            return self.fc(x).squeeze(-1)

        if self.pool_mode == 'mean':
            denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            mean_vec = (bert_out * attention_mask.unsqueeze(-1)).sum(dim=1) / denom
            x = self.drop(mean_vec)
            return self.fc(x).squeeze(-1)

        # token_bilstm → convs → pool
        lstm_out, _ = self.lstm(bert_out)                   # (B, T, 2*lstm_hidden)
        x = lstm_out.permute(0, 2, 1)                       # (B, C, T)
        conv_outs = []
        for conv in self.convs:
            c = self.relu(conv(x))                          # (B, Ck, T)
            p = self.pool(c).squeeze(-1)                    # (B, Ck)
            conv_outs.append(p)
        x = torch.cat(conv_outs, dim=1)                     # (B, sum(Ck))
        x = self.drop(x)
        return self.fc(x).squeeze(-1)

# -------------------------
# Measurement utilities (GPU-aware)
# -------------------------
def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def measure_latency_ms(model, device, input_batch, repeats=200, warmup=10):
    model.to(device).eval()
    input_batch = {k: v.to(device) for k, v in input_batch.items()}
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**input_batch)
        _cuda_sync()
        t0 = time.time()
        for _ in range(repeats):
            _ = model(**input_batch)
        _cuda_sync()
        elapsed = time.time() - t0
    return (elapsed / repeats) * 1000.0

def measure_throughput(model, device, batch_tensor, repeats=30, warmup=3):
    model.to(device).eval()
    batch = {k: v.to(device) for k, v in batch_tensor.items()}
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**batch)
        _cuda_sync()
        t0 = time.time()
        for _ in range(repeats):
            _ = model(**batch)
        _cuda_sync()
        elapsed = time.time() - t0
    bs = batch[next(iter(batch))].shape[0]
    return (bs * repeats) / max(elapsed, 1e-9)

def measure_peak_vram_mb(model, device, batch_tensor):
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device).eval()
    batch = {k: v.to(device) for k, v in batch_tensor.items()}
    with torch.no_grad():
        _ = model(**batch)
    _cuda_sync()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)

# -------------------------
# Data loading & helpers
# -------------------------
def load_data(pkl_path, device):
    """Load tokenized splits and return dataloaders + BCEWithLogitsLoss(pos_weight)."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Labels are already normalized in tokenize step → do NOT flip here
    log("Label distribution (train/val):")
    log(f"  Train: {Counter(data['train']['labels'])}")
    log(f"  Val  : {Counter(data['test']['labels'])}")

    train_ds = CodeDataset(data['train']['inputs'], data['train']['labels'])
    val_ds   = CodeDataset(data['test']['inputs'],  data['test']['labels'])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=PIN_MEMORY, num_workers=TRAIN_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=PIN_MEMORY, num_workers=VAL_WORKERS
    )

    labels_np = np.array(data['train']['labels'], dtype=int)
    try:
        classes   = np.unique(labels_np)
        cw        = compute_class_weight('balanced', classes=classes, y=labels_np)
        w_by_cls  = {int(c): float(w) for c, w in zip(classes, cw)}
        pos_w_val = w_by_cls.get(1, 1.0) / max(w_by_cls.get(0, 1.0), 1e-9)
    except Exception:
        w_by_cls  = {0: 1.0, 1: 1.0}
        pos_w_val = 1.0

    pos_w = torch.tensor(pos_w_val, dtype=torch.float, device=device)
    log(f"Class weights → pos_weight (class 1): {pos_w.item():.4f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    return train_loader, val_loader, loss_fn, len(train_ds), len(val_ds)

# -------------------------
# Training / Evaluation
# -------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids = batch['input_ids'].to(device)
        am  = batch['attention_mask'].to(device)
        lb  = batch['label'].to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                logits = model(ids, am)
                loss   = loss_fn(logits, lb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(ids, am)
            loss   = loss_fn(logits, lb)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * ids.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_probs_and_metrics(model, loader, device, loss_fn=None):
    """Return probs (sigmoid logits), labels, and average loss (if loss_fn provided)."""
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    for batch in loader:
        ids = batch['input_ids'].to(device)
        am  = batch['attention_mask'].to(device)
        lb  = batch['label'].to(device)

        logits = model(ids, am)
        probs  = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(lb.long().cpu().numpy().tolist())

        if loss_fn is not None:
            total_loss += loss_fn(logits, lb).item() * ids.size(0)

    avg_loss = total_loss / len(loader.dataset) if loss_fn is not None else None
    return np.array(all_probs), np.array(all_labels), avg_loss

def compute_binary_metrics(labels, probs, threshold=0.5):
    """Compute standard binary metrics at a given threshold."""
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    try:
        roc = roc_auc_score(labels, probs)
    except Exception:
        roc = float('nan')
    brier = brier_score_loss(labels, probs)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc, "brier": brier}

# -------------------------
# Main
# -------------------------
def main():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device} | BACKBONE={BACKBONE} | POOL_MODE={POOL_MODE} | BASE={BASE} | AMP={AMP_ENABLED}")

    tokenized_pkl = f'data/{BASE}_tokenized.pkl'
    if not os.path.exists(tokenized_pkl):
        raise FileNotFoundError(f"Tokenized data not found: {tokenized_pkl}  (Run tokenize_CodeBERT.py once and reuse)")

    # ---- Safety check: ensure labels were normalized in tokenize step (manifest) ----
    manifest_path = f"data/{BASE}_tokenized_manifest.json"
    if os.path.exists(manifest_path):
        try:
            m = json.load(open(manifest_path, "r", encoding="utf-8"))
            assert m.get("normalized_at") == "tokenize", \
                "Labels must be normalized exactly once in the tokenize step."
        except Exception as e:
            log(f"[WARN] Could not verify tokenized manifest: {e}")
    else:
        log("[WARN] tokenized manifest not found; proceeding without normalization assertion.")

    train_loader, val_loader, loss_fn, n_train, n_val = load_data(tokenized_pkl, device)

    # Build model with env-driven hyperparams
    model = CodeBERT_BiLSTM_CNN(
        model_name=BACKBONE,
        lstm_hidden=LSTM_HIDDEN,
        cnn_channels=CNN_FILTERS,
        cnn_kernels=CNN_KERNELS,
        dropout_p=DROPOUT,
        pool_mode=POOL_MODE
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Param count: {param_count:,}")

    scaler = torch.amp.GradScaler(enabled=(AMP_ENABLED and torch.cuda.is_available()))

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    best_f1, best_epoch = 0.0, 0
    epochs_no_improve = 0
    train_losses, val_losses, f1_scores = [], [], []

    total_training_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ------------- Training loop -------------
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_probs, val_labels, val_loss = evaluate_probs_and_metrics(model, val_loader, device, loss_fn)

        # Early stopping monitoring at fixed threshold=0.5 (simple & deterministic)
        metrics = compute_binary_metrics(val_labels, val_probs, threshold=0.5)
        val_f1 = metrics["f1"]

        epoch_time = time.time() - epoch_start
        train_losses.append(tr_loss)
        val_losses.append(val_loss if val_loss is not None else 0.0)
        f1_scores.append(val_f1)

        improved = val_f1 > best_f1
        if (epoch % LOG_EVERY == 0) or improved:
            print(f"Epoch {epoch}/{EPOCHS} | {epoch_time:.2f}s | "
                  f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1@0.5: {val_f1:.4f}")

        if improved:
            best_f1, best_epoch = val_f1, epoch
            torch.save(model.state_dict(), f'checkpoints/best_{BASE}.pt')
            epochs_no_improve = 0
            # Save validation probs/labels at best
            try:
                np.save(f"results/{BASE}_val_probs.npy",  val_probs)
                np.save(f"results/{BASE}_val_labels.npy", val_labels)
            except Exception as e:
                log(f"[WARN] could not save val probs: {e}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP:
            print(f"\nEarly stopping after {epoch} epochs (no improvement in {EARLY_STOP} epochs).")
            break

    total_training_time = time.time() - total_training_start
    peak_memory_mb = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
    log(f"Training time (s): {total_training_time:.2f} | peak GPU mem MB: {peak_memory_mb:.2f} | best_val_f1@0.5: {best_f1:.4f} (epoch {best_epoch})")

    # Save last checkpoint
    try:
        torch.save(model.state_dict(), f'checkpoints/{BASE}_last.pt')
    except Exception as e:
        log(f"[WARN] saving last checkpoint failed: {e}")

    # Small plots (optional)
    try:
        if not SKIP_PLOTS:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            epochs_axis = list(range(1, len(train_losses) + 1))

            plt.figure(figsize=(9,4))
            plt.plot(epochs_axis, train_losses, label='train_loss')
            plt.plot(epochs_axis, val_losses,   label='val_loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig(f'plots/loss_curve_{BASE}.png', dpi=300, bbox_inches='tight'); plt.close()

            plt.figure(figsize=(9,4))
            plt.plot(epochs_axis, f1_scores, label='val_f1@0.5')
            plt.xlabel('Epoch'); plt.ylabel('F1'); plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig(f'plots/f1_curve_{BASE}.png', dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e:
        log(f"[WARN] plotting skipped: {e}")

    # Compute validation ROC-AUC and BEST threshold from PR curve (for RQ3.1)
    try:
        val_probs_all = np.load(f"results/{BASE}_val_probs.npy")
        val_labels_all = np.load(f"results/{BASE}_val_labels.npy")
        try:
            auc_score = roc_auc_score(val_labels_all, val_probs_all)
        except Exception:
            auc_score = float('nan')

        # Best threshold by maximizing F1 over PR curve
        precs, recs, thrs = precision_recall_curve(val_labels_all, val_probs_all)
        f1s = 2 * precs * recs / (precs + recs + 1e-9)
        if len(thrs) > 0 and len(f1s) > 1:
            best_thr_idx = int(np.argmax(f1s[1:]))  # skip index 0 (no-threshold)
            best_thr = float(thrs[best_thr_idx])
            best_val_f1_at_thr = float(np.max(f1s[1:]))
        else:
            best_thr = 0.5
            best_val_f1_at_thr = float('nan')

        # Persist best threshold for test-time reuse
        with open(f'results/best_threshold_{BASE}.txt', 'w') as fh:
            fh.write(str(best_thr))

    except Exception:
        val_probs_all, val_labels_all, auc_score = None, None, float('nan')
        best_thr, best_val_f1_at_thr = 0.5, float('nan')

    # Human-readable TXT summary
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    with open(f"results/training_summary_{BASE}.txt", "w") as f:
        f.write(f"mode={MODE}\n")
        f.write(f"base={BASE}\n")
        f.write(f"backbone={BACKBONE}\n")
        f.write(f"seed={SEED}\n")
        f.write(f"pool_mode={POOL_MODE}\n")
        f.write(f"lstm_hidden={LSTM_HIDDEN}\n")
        f.write(f"cnn_filters={CNN_FILTERS}\n")
        f.write(f"cnn_kernels={CNN_KERNELS}\n")
        f.write(f"dropout={DROPOUT}\n")
        f.write(f"batch_size={BATCH_SIZE}\n")
        f.write(f"epochs_run={len(f1_scores)}\n")
        f.write(f"lr={LR}\n")
        f.write(f"param_count={param_count}\n")
        f.write(f"device={device_name}\n")
        f.write(f"train_size={n_train}\n")
        f.write(f"val_size={n_val}\n")
        f.write(f"total_training_time_s={total_training_time:.2f}\n")
        f.write(f"peak_gpu_mem_mb={peak_memory_mb:.2f}\n")
        f.write(f"best_val_f1_at_0.5={best_f1:.4f}\n")
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"val_roc_auc={auc_score}\n")
        f.write(f"best_val_threshold={best_thr}\n")
        f.write(f"best_val_f1_at_threshold={best_val_f1_at_thr}\n")

    # Machine-readable JSON summary (with compatibility keys for summarize_results.py)
    train_json = {
        "mode": MODE,
        "base": BASE,
        "backbone": BACKBONE,
        "seed": SEED,
        "pool_mode": POOL_MODE,
        "lstm_hidden": LSTM_HIDDEN,
        "cnn_filters": CNN_FILTERS,
        "cnn_kernels": CNN_KERNELS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs_run": int(len(f1_scores)),
        "lr": LR,
        "param_count": int(param_count),
        "device": device_name,
        "train_size": int(n_train),
        "val_size": int(n_val),
        "total_training_time_s": float(total_training_time),
        "peak_gpu_mem_mb": float(peak_memory_mb),
        "best_val_f1_at_0.5": float(best_f1),
        "best_epoch": int(best_epoch),
        "val_roc_auc": float(auc_score) if not np.isnan(auc_score) else None,
        "best_threshold_val": float(best_thr),
        "best_val_f1_at_threshold": float(best_val_f1_at_thr) if not np.isnan(best_val_f1_at_thr) else None,
    }
    train_json["timestamp"] = datetime.now(timezone.utc).isoformat()
    # ---- Compatibility keys (so summarize_results.py can read without changes) ----
    train_json["best_val_f1"] = (
        train_json.get("best_val_f1_at_threshold") 
        if train_json.get("best_val_f1_at_threshold") is not None 
        else train_json.get("best_val_f1_at_0.5")
    )
    train_json["best_threshold"] = train_json.get("best_threshold_val", 0.5)

    try:
        with open(f"results/training_summary_{BASE}.json", "w", encoding="utf-8") as jf:
            json.dump(train_json, jf, indent=2)
        log(f"Wrote results/training_summary_{BASE}.json")
    except Exception as e:
        log(f"[WARN] could not write JSON training summary: {e}")

    # Optional: evaluate best checkpoint on the FINAL test split and write final_test_results
    if RUN_TEST_AFTER:
        try:
            with open(tokenized_pkl, 'rb') as f:
                data = pickle.load(f)
            # Use FINAL split for test-time evaluation
            test_inputs = data['finaltest']['inputs']
            test_labels = data['finaltest']['labels']  # already normalized in tokenize step
            test_ds = CodeDataset(test_inputs, test_labels)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     pin_memory=PIN_MEMORY, num_workers=VAL_WORKERS)

            # Load best checkpoint if available
            ckpt_path = f'checkpoints/best_{BASE}.pt'
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
            else:
                log(f"[WARN] best checkpoint not found at {ckpt_path} — using final model state")

            # Compute test probabilities
            test_probs, test_labels_arr, _ = evaluate_probs_and_metrics(model, test_loader, device, None)

            # Use best validation threshold if available; fall back to 0.5
            try:
                with open(f"results/best_threshold_{BASE}.txt", "r") as fh:
                    val_thr = float(fh.read().strip())
            except Exception:
                val_thr = 0.5

            # Compute metrics at that threshold
            metrics_test = compute_binary_metrics(test_labels_arr, test_probs, threshold=val_thr)

            # Optional inference efficiency (GPU only)
            latency_ms = throughput = peak_vram = None
            if MEASURE_INFERENCE and torch.cuda.is_available():
                sample = next(iter(test_loader))
                # Latency on batch=1
                input1 = {
                    'input_ids':      sample['input_ids'][0].unsqueeze(0),
                    'attention_mask': sample['attention_mask'][0].unsqueeze(0)
                }
                latency_ms = measure_latency_ms(model, device, input1, repeats=200, warmup=10)

                # Throughput on a small batch (up to 32 samples)
                bs_thr = min(32, sample['input_ids'].shape[0])
                ids_rep = sample['input_ids'][:bs_thr]
                am_rep  = sample['attention_mask'][:bs_thr]
                batch_tensor = {'input_ids': ids_rep, 'attention_mask': am_rep}
                throughput = measure_throughput(model, device, batch_tensor, repeats=30, warmup=3)
                peak_vram  = measure_peak_vram_mb(model, device, batch_tensor)

            # Write final test results
            with open(f"results/final_test_results_{BASE}.txt", "w") as f:
                f.write(f"mode={MODE}\n")
                f.write(f"base={BASE}\n")
                f.write(f"backbone={BACKBONE}\n")
                f.write(f"seed={SEED}\n")
                f.write(f"threshold_used={val_thr}\n")
                f.write(f"f1={metrics_test['f1']:.4f}\n")
                f.write(f"precision={metrics_test['precision']:.4f}\n")
                f.write(f"recall={metrics_test['recall']:.4f}\n")
                f.write(f"roc_auc={metrics_test['roc_auc']:.4f}\n")
                f.write(f"brier={metrics_test['brier']:.6f}\n")
                if latency_ms is not None:
                    f.write(f"latency_ms_per_sample={latency_ms:.3f}\n")
                if throughput is not None:
                    f.write(f"throughput_samples_per_sec={throughput:.3f}\n")
                if peak_vram is not None:
                    f.write(f"peak_gpu_mem_mb={peak_vram:.2f}\n")

            # Save test arrays
            try:
                np.save(f"results/{BASE}_test_probs.npy",  test_probs)
                np.save(f"results/{BASE}_test_labels.npy", test_labels_arr)
            except Exception as e:
                log(f"[WARN] saving test probs failed: {e}")

            log(f"Wrote final_test_results_{BASE}.txt (F1={metrics_test['f1']:.4f})")
        except Exception as e:
            log(f"[WARN] final test evaluation failed: {e}")

if __name__ == "__main__":
    main()
