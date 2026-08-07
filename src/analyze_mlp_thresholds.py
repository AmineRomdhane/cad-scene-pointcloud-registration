#!/usr/bin/env python3
"""
Analyze different classification thresholds for the weighted MLP.

This script retrains the reduced-feature MLP using the same leave-one-shape-out
protocol, then evaluates several thresholds instead of only 0.5.

Outputs:
- results/tables/mlp_threshold_results_by_shape.csv
- results/tables/mlp_threshold_results_average.csv
- results/figures/mlp_threshold_average_curves.png
- results/figures/mlp_threshold_confusion_matrix_<shape>_thr_<threshold>.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import torch
import torch.nn as nn

from train_mlp_pytorch import (
    BASE_DIR,
    INPUT_CSV,
    TABLE_DIR,
    FIGURE_DIR,
    TARGET,
    REDUCED_FEATURES,
    KNOWN_SHAPES,
    RANDOM_STATE,
    EPOCHS,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    PATIENCE,
    set_seed,
    infer_shape_name,
    CorrespondenceMLP,
    make_loader,
    run_epoch,
    predict_proba,
    split_train_val_by_sample_id,
    plot_confusion_matrix,
)


THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def compute_threshold_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": safe_auc(y_true, y_proba),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "num_predicted_reliable": int(np.sum(y_pred == 1)),
        "acceptance_rate": float(np.mean(y_pred == 1)),
    }


def train_one_fold(df, test_shape, device):
    trainval_df = df[df["shape_name"] != test_shape].copy()
    test_df = df[df["shape_name"] == test_shape].copy()

    inner_train_df, val_df = split_train_val_by_sample_id(trainval_df)

    X_train_raw = inner_train_df[REDUCED_FEATURES].values.astype(np.float32)
    y_train = inner_train_df[TARGET].astype(int).values.astype(np.float32)

    X_val_raw = val_df[REDUCED_FEATURES].values.astype(np.float32)
    y_val = val_df[TARGET].astype(int).values.astype(np.float32)

    X_test_raw = test_df[REDUCED_FEATURES].values.astype(np.float32)
    y_test = test_df[TARGET].astype(int).values.astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))

    pos_weight_value = n_neg / max(n_pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    model = CorrespondenceMLP(input_dim=len(REDUCED_FEATURES)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = np.inf
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    print("\n" + "=" * 80)
    print(f"Training weighted reduced MLP | test_shape={test_shape}")
    print("=" * 80)
    print(f"Train rows: {len(inner_train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Test rows:  {len(test_df)}")
    print(f"pos_weight: {pos_weight_value:.4f}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={val_loss:.5f} | "
                f"best_epoch={best_epoch}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test_proba = predict_proba(model, X_test, device)

    return y_test.astype(int), y_test_proba, test_df, best_epoch, best_val_loss


def plot_average_threshold_curves(avg_df, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(avg_df["threshold"], avg_df["precision_mean"], marker="o", label="Precision")
    ax.plot(avg_df["threshold"], avg_df["recall_mean"], marker="o", label="Recall")
    ax.plot(avg_df["threshold"], avg_df["f1_mean"], marker="o", label="F1")
    ax.plot(avg_df["threshold"], avg_df["accuracy_mean"], marker="o", label="Accuracy")

    ax.set_title("Weighted MLP threshold analysis")
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def summarize_by_threshold(results_df):
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

    for threshold, group in results_df.groupby("threshold"):
        row = {
            "model": "MLP_weighted_reduced",
            "threshold": threshold,
            "num_folds": len(group),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std()

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(by="threshold")

    return out


def main():
    set_seed(RANDOM_STATE)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["sample_id", "scenario", TARGET] + REDUCED_FEATURES
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols)
    df[TARGET] = df[TARGET].astype(int)
    df["shape_name"] = df["sample_id"].apply(infer_shape_name)

    all_rows = []

    for test_shape in KNOWN_SHAPES:
        y_test, y_proba, test_df, best_epoch, best_val_loss = train_one_fold(
            df=df,
            test_shape=test_shape,
            device=device,
        )

        for threshold in THRESHOLDS:
            metrics = compute_threshold_metrics(
                y_true=y_test,
                y_proba=y_proba,
                threshold=threshold,
            )

            row = {
                "model": "MLP_weighted_reduced",
                "feature_set": "reduced",
                "test_shape": test_shape,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "test_rows": len(test_df),
                "test_positive_rate": float(np.mean(y_test)),
            }

            row.update(metrics)
            all_rows.append(row)

            y_pred = (y_proba >= threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

            # Save confusion matrices only for the most useful thresholds
            if threshold in [0.5, 0.6, 0.7, 0.8]:
                threshold_tag = str(threshold).replace(".", "p")
                cm_path = FIGURE_DIR / (
                    f"mlp_threshold_confusion_matrix_"
                    f"{test_shape}_thr_{threshold_tag}.png"
                )

                plot_confusion_matrix(
                    cm,
                    f"MLP weighted reduced | test={test_shape} | threshold={threshold}",
                    cm_path,
                )

    results_df = pd.DataFrame(all_rows)

    by_shape_path = TABLE_DIR / "mlp_threshold_results_by_shape.csv"
    results_df.to_csv(by_shape_path, index=False)

    avg_df = summarize_by_threshold(results_df)

    avg_path = TABLE_DIR / "mlp_threshold_results_average.csv"
    avg_df.to_csv(avg_path, index=False)

    curve_path = FIGURE_DIR / "mlp_threshold_average_curves.png"
    plot_average_threshold_curves(avg_df, curve_path)

    print("\n" + "=" * 80)
    print("Average threshold results:")
    print(avg_df.to_string(index=False))

    print("\nSaved:")
    print(f"  {by_shape_path}")
    print(f"  {avg_path}")
    print(f"  {curve_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
