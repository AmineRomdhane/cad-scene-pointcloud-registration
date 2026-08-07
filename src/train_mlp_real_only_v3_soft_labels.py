#!/usr/bin/env python3
"""
MLP real-only v3 with soft labels.

Dataset:
- results/learning_data_synthetic_plus_real_curated_clean_v3/all_correspondences.csv

Experiment:
- real data only
- leave-one-real-case-out evaluation
- soft target from label_distance
- same reduced feature set as before

soft_target = exp(-label_distance^2 / (2 * sigma^2))
"""

from pathlib import Path
import copy
import math
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
    mean_squared_error,
    mean_absolute_error,
)


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    BASE_DIR
    / "results"
    / "learning_data_synthetic_plus_real_curated_clean_v3"
    / "all_correspondences.csv"
)

TABLE_DIR = BASE_DIR / "results" / "tables"

MODEL_NAME = "MLP_real_only_v3_soft_labels"

FEATURES = [
    "distance_T0",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
]

HARD_TARGET = "target_weight"
LABEL_DISTANCE = "label_distance"
SOFT_TARGET = "soft_target"

SIGMA = 0.08

RANDOM_STATE = 42

EPOCHS = 120
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15

VAL_CASE_FRACTION = 0.20

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class CorrespondenceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.10),
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


def make_soft_target(label_distance, sigma):
    d = np.asarray(label_distance, dtype=np.float64)
    soft = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    return np.clip(soft, 0.0, 1.0)


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, score)


def safe_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 2:
        return np.nan

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])

    if method == "spearman":
        return float(pd.Series(x).corr(pd.Series(y), method="spearman"))

    raise ValueError(f"Unknown method: {method}")


def compute_hard_metrics(y_true, score, threshold):
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


def compute_soft_metrics(soft_true, score, label_distance):
    mse = mean_squared_error(soft_true, score)

    return {
        "soft_mse": float(mse),
        "soft_rmse": float(math.sqrt(mse)),
        "soft_mae": float(mean_absolute_error(soft_true, score)),
        "pearson_pred_soft_target": safe_corr(score, soft_true, "pearson"),
        "spearman_pred_soft_target": safe_corr(score, soft_true, "spearman"),
        "pearson_pred_negative_label_distance": safe_corr(score, -label_distance, "pearson"),
        "spearman_pred_negative_label_distance": safe_corr(score, -label_distance, "spearman"),
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
        for start in range(0, len(X_tensor), 4096):
            xb = X_tensor[start:start + 4096]
            logits = model(xb)
            prob = torch.sigmoid(logits)
            scores.append(prob.cpu().numpy())

    return np.concatenate(scores, axis=0)


def train_one_fold(train_df, val_df, test_df, heldout_case, device):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df[FEATURES].values)
    X_val = scaler.transform(val_df[FEATURES].values)
    X_test = scaler.transform(test_df[FEATURES].values)

    y_train_soft = train_df[SOFT_TARGET].values.astype(np.float32)
    y_val_soft = val_df[SOFT_TARGET].values.astype(np.float32)
    y_test_soft = test_df[SOFT_TARGET].values.astype(np.float32)

    y_test_hard = test_df[HARD_TARGET].values.astype(int)
    test_label_distance = test_df[LABEL_DISTANCE].values.astype(np.float64)

    train_loader = make_loader(X_train, y_train_soft, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val_soft, BATCH_SIZE, shuffle=False)

    model = CorrespondenceMLP(input_dim=len(FEATURES)).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
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

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        history_rows.append({
            "model": MODEL_NAME,
            "heldout_real_case": heldout_case,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_epoch_so_far": best_epoch,
            "best_val_loss_so_far": best_val_loss,
        })

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={val_loss:.5f} | "
                f"best_epoch={best_epoch}"
            )

        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_score = predict_score(model, X_test, device)

    soft_metrics = compute_soft_metrics(
        soft_true=y_test_soft,
        score=test_score,
        label_distance=test_label_distance,
    )

    hard_metrics_05 = compute_hard_metrics(
        y_true=y_test_hard,
        score=test_score,
        threshold=0.5,
    )

    result_row = {
        "model": MODEL_NAME,
        "heldout_real_case": heldout_case,
        "sigma": SIGMA,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_hard_positive_rate": float(train_df[HARD_TARGET].mean()),
        "val_hard_positive_rate": float(val_df[HARD_TARGET].mean()),
        "test_hard_positive_rate": float(test_df[HARD_TARGET].mean()),
        "train_soft_target_mean": float(train_df[SOFT_TARGET].mean()),
        "val_soft_target_mean": float(val_df[SOFT_TARGET].mean()),
        "test_soft_target_mean": float(test_df[SOFT_TARGET].mean()),
    }

    result_row.update(soft_metrics)
    result_row.update(hard_metrics_05)

    threshold_rows = []

    for threshold in THRESHOLDS:
        m = compute_hard_metrics(
            y_true=y_test_hard,
            score=test_score,
            threshold=threshold,
        )

        threshold_rows.append({
            "model": MODEL_NAME,
            "heldout_real_case": heldout_case,
            "sigma": SIGMA,
            **m,
        })

    return result_row, threshold_rows, history_rows


def summarize_average(df):
    metric_cols = [
        "soft_mse",
        "soft_rmse",
        "soft_mae",
        "pearson_pred_soft_target",
        "spearman_pred_soft_target",
        "pearson_pred_negative_label_distance",
        "spearman_pred_negative_label_distance",
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
        "sigma": SIGMA,
        "num_folds": len(df),
    }

    for col in metric_cols:
        row[f"{col}_mean"] = df[col].mean()
        row[f"{col}_std"] = df[col].std()

    return pd.DataFrame([row])


def summarize_thresholds(df):
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

    for threshold, group in df.groupby("threshold"):
        row = {
            "model": MODEL_NAME,
            "sigma": SIGMA,
            "num_folds": group["heldout_real_case"].nunique(),
            "threshold": threshold,
        }

        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std()

        rows.append(row)

    return pd.DataFrame(rows).sort_values("threshold")


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("MLP real-only v3 with soft labels")
    print("=" * 100)
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Features: {FEATURES}")
    print(f"SIGMA: {SIGMA}")
    print("=" * 100)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["sample_id", HARD_TARGET, LABEL_DISTANCE] + FEATURES
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols).copy()

    df["sample_id"] = df["sample_id"].astype(str)
    df = df[df["sample_id"].str.startswith("real_")].copy()

    df["real_case_id"] = df["sample_id"].apply(get_real_case_id)
    df[HARD_TARGET] = df[HARD_TARGET].astype(int)
    df[SOFT_TARGET] = make_soft_target(df[LABEL_DISTANCE].values, SIGMA)

    real_case_ids = sorted(df["real_case_id"].unique())

    print(f"Real rows:       {len(df)}")
    print(f"Real cases:      {len(real_case_ids)}")
    print(f"Hard pos rate:   {df[HARD_TARGET].mean():.4f}")
    print(f"Soft mean:       {df[SOFT_TARGET].mean():.4f}")
    print("-" * 100)

    if len(real_case_ids) < 3:
        raise RuntimeError("Need at least 3 real cases for train/val/test split.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("-" * 100)

    result_rows = []
    threshold_rows_all = []
    history_rows_all = []

    for heldout_index, heldout_case in enumerate(real_case_ids):
        print("\n" + "=" * 100)
        print(f"Held-out real case: {heldout_case}")
        print("=" * 100)

        test_df = df[df["real_case_id"] == heldout_case].copy()
        trainval_df = df[df["real_case_id"] != heldout_case].copy()

        available_cases = sorted(trainval_df["real_case_id"].unique())

        num_val_cases = max(1, int(round(VAL_CASE_FRACTION * len(available_cases))))

        rng = np.random.default_rng(RANDOM_STATE + heldout_index)

        val_cases = set(
            rng.choice(
                available_cases,
                size=num_val_cases,
                replace=False,
            ).tolist()
        )

        val_df = trainval_df[trainval_df["real_case_id"].isin(val_cases)].copy()
        train_df = trainval_df[~trainval_df["real_case_id"].isin(val_cases)].copy()

        print(f"Train rows:       {len(train_df)}")
        print(f"Val rows:         {len(val_df)}")
        print(f"Test rows:        {len(test_df)}")
        print(f"Train hard pos:   {train_df[HARD_TARGET].mean():.4f}")
        print(f"Val hard pos:     {val_df[HARD_TARGET].mean():.4f}")
        print(f"Test hard pos:    {test_df[HARD_TARGET].mean():.4f}")
        print(f"Train soft mean:  {train_df[SOFT_TARGET].mean():.4f}")
        print(f"Val soft mean:    {val_df[SOFT_TARGET].mean():.4f}")
        print(f"Test soft mean:   {test_df[SOFT_TARGET].mean():.4f}")

        result_row, threshold_rows, history_rows = train_one_fold(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            heldout_case=heldout_case,
            device=device,
        )

        result_rows.append(result_row)
        threshold_rows_all.extend(threshold_rows)
        history_rows_all.extend(history_rows)

        print(
            f"Result | "
            f"AUC={result_row['roc_auc']:.3f} | "
            f"F1@0.5={result_row['f1']:.3f} | "
            f"precision={result_row['precision']:.3f} | "
            f"recall={result_row['recall']:.3f} | "
            f"soft_RMSE={result_row['soft_rmse']:.3f} | "
            f"spearman_soft={result_row['spearman_pred_soft_target']:.3f}"
        )

    results_df = pd.DataFrame(result_rows)
    threshold_df = pd.DataFrame(threshold_rows_all)
    history_df = pd.DataFrame(history_rows_all)

    avg_df = summarize_average(results_df)
    threshold_avg_df = summarize_thresholds(threshold_df)

    results_path = TABLE_DIR / "mlp_real_only_v3_soft_labels_results_by_case.csv"
    avg_path = TABLE_DIR / "mlp_real_only_v3_soft_labels_results_average.csv"
    threshold_path = TABLE_DIR / "mlp_real_only_v3_soft_labels_threshold_results_by_case.csv"
    threshold_avg_path = TABLE_DIR / "mlp_real_only_v3_soft_labels_threshold_results_average.csv"
    history_path = TABLE_DIR / "mlp_real_only_v3_soft_labels_training_history.csv"

    results_df.to_csv(results_path, index=False)
    avg_df.to_csv(avg_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    threshold_avg_df.to_csv(threshold_avg_path, index=False)
    history_df.to_csv(history_path, index=False)

    print("\n" + "=" * 100)
    print("Average result:")
    print(avg_df.to_string(index=False))

    print("\nThreshold average:")
    print(threshold_avg_df.to_string(index=False))

    print("\nSaved:")
    print(f"  {results_path}")
    print(f"  {avg_path}")
    print(f"  {threshold_path}")
    print(f"  {threshold_avg_path}")
    print(f"  {history_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
