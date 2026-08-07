#!/usr/bin/env python3
"""
Train MLP on:
- all synthetic samples
- only 2 selected real v3 cases

Test on:
- all remaining real v3 cases, one by one

This is a real-data dose experiment:
How much does adding only two curated real cases to synthetic training help
on unseen real cases?

This script intentionally rolls back to the older stable MLP setup:
- hard 0/1 labels
- reduced features
- dropout = 0.10
- LR = 1e-3
- automatic pos_weight
- early stopping on validation loss
"""

from pathlib import Path
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    BASE_DIR
    / "results"
    / "learning_data_synthetic_plus_real_curated_clean_v3"
    / "all_correspondences.csv"
)

TEST_NAME = "mlp_synthetic_plus_2real_v3_test_others"
MODEL_NAME = "MLP_synthetic_plus_2real_v3_test_others"

OUT_DIR = BASE_DIR / "results" / "by_test" / TEST_NAME

FEATURES = [
    "distance_T0",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
]

TARGET = "target_weight"

TRAIN_REAL_CASES = [
    "20260626_144015_pallet_rack_05_part1_refined_from_voxel005",
    "20260706_130424_table_cluster004_table1_refined_from_identity_scale1",
]

RANDOM_STATE = 42

LR = 1e-3
DROPOUT = 0.10
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1024
EPOCHS = 150
PATIENCE = 15

VAL_FRACTION_BY_SAMPLE_ID = 0.20

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class CorrespondenceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def get_real_case_id(sample_id):
    sample_id = str(sample_id)

    if not sample_id.startswith("real_"):
        return ""

    case_id = sample_id[len("real_"):]

    for suffix in ["_easy", "_medium", "_hard"]:
        if case_id.endswith(suffix):
            case_id = case_id[: -len(suffix)]
            break

    return case_id


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, score)


def compute_metrics(y_true, score, threshold):
    pred = (score >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": safe_auc(y_true, score),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "num_predicted_reliable": int(np.sum(pred == 1)),
        "acceptance_rate": float(np.mean(pred == 1)),
    }


def make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def predict_score(model, X, device):
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    scores = []

    with torch.no_grad():
        for start in range(0, len(X_tensor), 8192):
            xb = X_tensor[start:start + 8192]
            logits = model(xb)
            prob = torch.sigmoid(logits)
            scores.append(prob.cpu().numpy())

    return np.concatenate(scores, axis=0)


def split_train_validation_by_sample_id(train_pool_df):
    unique_sample_ids = np.array(sorted(train_pool_df["sample_id"].unique()))

    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_sample_ids)

    n_val = max(1, int(round(VAL_FRACTION_BY_SAMPLE_ID * len(unique_sample_ids))))

    val_sample_ids = set(unique_sample_ids[:n_val].tolist())

    val_df = train_pool_df[train_pool_df["sample_id"].isin(val_sample_ids)].copy()
    train_df = train_pool_df[~train_pool_df["sample_id"].isin(val_sample_ids)].copy()

    return train_df, val_df, sorted(val_sample_ids)


def train_model(train_df, val_df, device):
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES].values)

    X_train = scaler.transform(train_df[FEATURES].values)
    y_train = train_df[TARGET].values.astype(np.float32)

    X_val = scaler.transform(val_df[FEATURES].values)
    y_val = val_df[TARGET].values.astype(np.float32)

    num_pos = float(np.sum(y_train == 1))
    num_neg = float(np.sum(y_train == 0))

    if num_pos <= 0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = num_neg / num_pos

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    model = CorrespondenceMLP(input_dim=len(FEATURES)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = make_loader(
        X_train,
        y_train,
        BATCH_SIZE,
        shuffle=True,
    )

    val_loader = make_loader(
        X_val,
        y_val,
        BATCH_SIZE,
        shuffle=False,
    )

    best_val_loss = np.inf
    best_epoch = 0
    best_state = None
    patience_count = 0

    history_rows = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))
        val_score = predict_score(model, X_val, device)
        val_auc = safe_auc(y_val.astype(int), val_score)

        improved = val_loss < best_val_loss - 1e-6

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        history_rows.append({
            "model": MODEL_NAME,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "best_epoch_so_far": best_epoch,
            "best_val_loss_so_far": best_val_loss,
            "patience_count": patience_count,
            "pos_weight": pos_weight_value,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "train_positive_rate": float(np.mean(y_train)),
            "val_positive_rate": float(np.mean(y_val)),
        })

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={val_loss:.5f} | "
                f"val_auc={val_auc:.4f} | "
                f"best_epoch={best_epoch}"
            )

        if patience_count >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch = {best_epoch}, best val loss = {best_val_loss:.6f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_info = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "pos_weight": pos_weight_value,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_positive_rate": float(np.mean(y_train)),
        "val_positive_rate": float(np.mean(y_val)),
    }

    return model, scaler, pd.DataFrame(history_rows), train_info


def evaluate_on_real_cases(model, scaler, df, test_real_cases, device, train_info):
    result_rows = []
    threshold_rows = []

    for case_id in test_real_cases:
        test_df = df[df["real_case_id"] == case_id].copy()

        if len(test_df) == 0:
            continue

        X_test = scaler.transform(test_df[FEATURES].values)
        y_test = test_df[TARGET].values.astype(int)

        score = predict_score(model, X_test, device)

        metrics_05 = compute_metrics(
            y_true=y_test,
            score=score,
            threshold=0.5,
        )

        result_row = {
            "model": MODEL_NAME,
            "heldout_real_case": case_id,
            "num_test_rows": len(test_df),
            "test_positive_rate": float(np.mean(y_test)),
            "lr": LR,
            "dropout": DROPOUT,
            "best_epoch": train_info["best_epoch"],
            "best_val_loss": train_info["best_val_loss"],
            "pos_weight": train_info["pos_weight"],
            "train_rows": train_info["train_rows"],
            "val_rows": train_info["val_rows"],
            "train_positive_rate": train_info["train_positive_rate"],
            "val_positive_rate": train_info["val_positive_rate"],
        }

        result_row.update(metrics_05)
        result_rows.append(result_row)

        for threshold in THRESHOLDS:
            m = compute_metrics(
                y_true=y_test,
                score=score,
                threshold=threshold,
            )

            threshold_rows.append({
                "model": MODEL_NAME,
                "heldout_real_case": case_id,
                "threshold": threshold,
                "lr": LR,
                "dropout": DROPOUT,
                "best_epoch": train_info["best_epoch"],
                "pos_weight": train_info["pos_weight"],
                **m,
            })

        print(
            f"Test case: {case_id} | "
            f"AUC={metrics_05['roc_auc']:.3f} | "
            f"F1@0.5={metrics_05['f1']:.3f} | "
            f"precision={metrics_05['precision']:.3f} | "
            f"recall={metrics_05['recall']:.3f} | "
            f"acceptance={metrics_05['acceptance_rate']:.3f}"
        )

    return pd.DataFrame(result_rows), pd.DataFrame(threshold_rows)


def summarize_results(results_df):
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "fp",
        "fn",
        "tp",
        "tn",
        "num_predicted_reliable",
        "acceptance_rate",
    ]

    row = {
        "model": MODEL_NAME,
        "num_test_cases": results_df["heldout_real_case"].nunique(),
        "lr": LR,
        "dropout": DROPOUT,
        "train_real_cases": "|".join(TRAIN_REAL_CASES),
    }

    for col in metric_cols:
        row[f"{col}_mean"] = results_df[col].mean()
        row[f"{col}_std"] = results_df[col].std()

    return pd.DataFrame([row])


def summarize_thresholds(threshold_df):
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "fp",
        "fn",
        "tp",
        "tn",
        "num_predicted_reliable",
        "acceptance_rate",
    ]

    rows = []

    for threshold, group in threshold_df.groupby("threshold"):
        row = {
            "model": MODEL_NAME,
            "num_test_cases": group["heldout_real_case"].nunique(),
            "threshold": threshold,
            "lr": LR,
            "dropout": DROPOUT,
            "train_real_cases": "|".join(TRAIN_REAL_CASES),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std()

        rows.append(row)

    return pd.DataFrame(rows).sort_values("threshold")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    print("=" * 100)
    print(MODEL_NAME)
    print("=" * 100)
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Features: {FEATURES}")
    print(f"Train real cases:")
    for c in TRAIN_REAL_CASES:
        print(f"  - {c}")
    print("=" * 100)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["sample_id", TARGET] + FEATURES
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols).copy()

    df["sample_id"] = df["sample_id"].astype(str)
    df["is_real"] = df["sample_id"].str.startswith("real_")
    df["real_case_id"] = df["sample_id"].apply(get_real_case_id)
    df[TARGET] = df[TARGET].astype(int)

    real_case_ids = sorted(df[df["is_real"]]["real_case_id"].unique())

    missing_train_cases = [c for c in TRAIN_REAL_CASES if c not in real_case_ids]

    if missing_train_cases:
        print("Available real cases:")
        for c in real_case_ids:
            print(f"  - {c}")
        raise RuntimeError(f"Selected train cases not found: {missing_train_cases}")

    test_real_cases = [c for c in real_case_ids if c not in TRAIN_REAL_CASES]

    synthetic_df = df[~df["is_real"]].copy()
    train_real_df = df[df["real_case_id"].isin(TRAIN_REAL_CASES)].copy()

    train_pool_df = pd.concat(
        [synthetic_df, train_real_df],
        axis=0,
        ignore_index=True,
    )

    train_df, val_df, val_sample_ids = split_train_validation_by_sample_id(train_pool_df)

    print(f"Total rows after cleaning: {len(df)}")
    print(f"Synthetic rows:           {len(synthetic_df)}")
    print(f"Selected real train rows: {len(train_real_df)}")
    print(f"Train pool rows:          {len(train_pool_df)}")
    print(f"Train rows:               {len(train_df)}")
    print(f"Validation rows:          {len(val_df)}")
    print(f"Real cases in v3:         {len(real_case_ids)}")
    print(f"Train real cases:         {len(TRAIN_REAL_CASES)}")
    print(f"Test real cases:          {len(test_real_cases)}")
    print(f"Train positive rate:      {train_df[TARGET].mean():.4f}")
    print(f"Val positive rate:        {val_df[TARGET].mean():.4f}")
    print("-" * 100)

    print("Test real cases:")
    for c in test_real_cases:
        print(f"  - {c}")
    print("-" * 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("-" * 100)

    model, scaler, history_df, train_info = train_model(
        train_df=train_df,
        val_df=val_df,
        device=device,
    )

    results_df, threshold_df = evaluate_on_real_cases(
        model=model,
        scaler=scaler,
        df=df,
        test_real_cases=test_real_cases,
        device=device,
        train_info=train_info,
    )

    avg_df = summarize_results(results_df)
    threshold_avg_df = summarize_thresholds(threshold_df)

    train_real_cases_df = pd.DataFrame({
        "train_real_case": TRAIN_REAL_CASES,
    })

    split_summary_df = pd.DataFrame([{
        "model": MODEL_NAME,
        "input_csv": str(INPUT_CSV),
        "num_synthetic_rows": len(synthetic_df),
        "num_train_real_rows": len(train_real_df),
        "num_train_pool_rows": len(train_pool_df),
        "num_train_rows": len(train_df),
        "num_val_rows": len(val_df),
        "num_real_cases_v3": len(real_case_ids),
        "num_train_real_cases": len(TRAIN_REAL_CASES),
        "num_test_real_cases": len(test_real_cases),
        "train_real_cases": "|".join(TRAIN_REAL_CASES),
        "test_real_cases": "|".join(test_real_cases),
        "val_sample_ids": "|".join(val_sample_ids),
        "lr": LR,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "best_epoch": train_info["best_epoch"],
        "best_val_loss": train_info["best_val_loss"],
        "pos_weight": train_info["pos_weight"],
    }])

    results_path = OUT_DIR / f"{TEST_NAME}_results_by_case.csv"
    avg_path = OUT_DIR / f"{TEST_NAME}_results_average.csv"
    threshold_path = OUT_DIR / f"{TEST_NAME}_threshold_results_by_case.csv"
    threshold_avg_path = OUT_DIR / f"{TEST_NAME}_threshold_results_average.csv"
    history_path = OUT_DIR / f"{TEST_NAME}_training_history.csv"
    split_path = OUT_DIR / f"{TEST_NAME}_split_summary.csv"
    train_cases_path = OUT_DIR / f"{TEST_NAME}_train_real_cases.csv"

    results_df.to_csv(results_path, index=False)
    avg_df.to_csv(avg_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    threshold_avg_df.to_csv(threshold_avg_path, index=False)
    history_df.to_csv(history_path, index=False)
    split_summary_df.to_csv(split_path, index=False)
    train_real_cases_df.to_csv(train_cases_path, index=False)

    print("\n" + "=" * 100)
    print("Average over held-out real cases:")
    print(avg_df.to_string(index=False))

    print("\nThreshold average over held-out real cases:")
    print(threshold_avg_df.to_string(index=False))

    print("\nSaved:")
    print(f"  {results_path}")
    print(f"  {avg_path}")
    print(f"  {threshold_path}")
    print(f"  {threshold_avg_path}")
    print(f"  {history_path}")
    print(f"  {split_path}")
    print(f"  {train_cases_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
