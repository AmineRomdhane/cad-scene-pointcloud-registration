#!/usr/bin/env python3
"""
Train the reduced-feature MLP with balanced training data.

Important:
- Only the training split is balanced.
- Validation and test splits keep the original class distribution.
- Balancing is done by random undersampling of the majority class.
- No pos_weight is used because the training data is already balanced.

Outputs:
- results/tables/mlp_balanced_undersampling_results_by_shape.csv
- results/tables/mlp_balanced_undersampling_results_average.csv
- results/tables/mlp_balanced_undersampling_threshold_results_by_shape.csv
- results/tables/mlp_balanced_undersampling_threshold_results_average.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from train_mlp_pytorch import (
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


def balance_training_data_undersampling(train_df, target_col, random_state):
    """
    Balance only the training data by undersampling the majority class.

    If there are more bad correspondences than reliable correspondences,
    bad correspondences are randomly reduced.

    If there are more reliable correspondences than bad correspondences,
    reliable correspondences are randomly reduced.
    """

    pos_df = train_df[train_df[target_col] == 1].copy()
    neg_df = train_df[train_df[target_col] == 0].copy()

    n_pos = len(pos_df)
    n_neg = len(neg_df)

    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"Cannot balance training data. n_pos={n_pos}, n_neg={n_neg}"
        )

    n_keep = min(n_pos, n_neg)

    pos_balanced = pos_df.sample(
        n=n_keep,
        replace=False,
        random_state=random_state,
    )

    neg_balanced = neg_df.sample(
        n=n_keep,
        replace=False,
        random_state=random_state,
    )

    balanced_df = pd.concat([pos_balanced, neg_balanced], axis=0)

    balanced_df = balanced_df.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)

    return balanced_df


def compute_metrics(y_true, y_proba, threshold):
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

    original_train_rows = len(inner_train_df)
    original_train_positive_rate = float(inner_train_df[TARGET].mean())

    balanced_train_df = balance_training_data_undersampling(
        train_df=inner_train_df,
        target_col=TARGET,
        random_state=RANDOM_STATE,
    )

    balanced_train_rows = len(balanced_train_df)
    balanced_train_positive_rate = float(balanced_train_df[TARGET].mean())

    X_train_raw = balanced_train_df[REDUCED_FEATURES].values.astype(np.float32)
    y_train = balanced_train_df[TARGET].astype(int).values.astype(np.float32)

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

    model = CorrespondenceMLP(input_dim=len(REDUCED_FEATURES)).to(device)

    # No pos_weight here because the training data is already balanced.
    criterion = nn.BCEWithLogitsLoss()

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
    print(f"Training balanced undersampling MLP | test_shape={test_shape}")
    print("=" * 80)
    print(f"Original train rows:       {original_train_rows}")
    print(f"Original train pos rate:   {original_train_positive_rate:.4f}")
    print(f"Balanced train rows:       {balanced_train_rows}")
    print(f"Balanced train pos rate:   {balanced_train_positive_rate:.4f}")
    print(f"Val rows:                  {len(val_df)}")
    print(f"Val positive rate:         {val_df[TARGET].mean():.4f}")
    print(f"Test rows:                 {len(test_df)}")
    print(f"Test positive rate:        {test_df[TARGET].mean():.4f}")
    print("pos_weight:                disabled")

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
    y_test_int = y_test.astype(int)

    return {
        "test_shape": test_shape,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "original_train_rows": original_train_rows,
        "balanced_train_rows": balanced_train_rows,
        "original_train_positive_rate": original_train_positive_rate,
        "balanced_train_positive_rate": balanced_train_positive_rate,
        "val_rows": len(val_df),
        "val_positive_rate": float(val_df[TARGET].mean()),
        "test_rows": len(test_df),
        "test_positive_rate": float(test_df[TARGET].mean()),
        "y_test": y_test_int,
        "y_test_proba": y_test_proba,
    }


def summarize_results_by_threshold(threshold_df):
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
            "model": "MLP_balanced_undersampling",
            "feature_set": "reduced",
            "threshold": threshold,
            "num_folds": len(group),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std()

        rows.append(row)

    return pd.DataFrame(rows).sort_values(by="threshold")


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
    df = df.dropna(subset=required_cols).copy()
    df[TARGET] = df[TARGET].astype(int)
    df["shape_name"] = df["sample_id"].apply(infer_shape_name)

    by_shape_rows = []
    threshold_rows = []

    for test_shape in KNOWN_SHAPES:
        fold = train_one_fold(
            df=df,
            test_shape=test_shape,
            device=device,
        )

        y_test = fold["y_test"]
        y_test_proba = fold["y_test_proba"]

        # Main result at default threshold 0.5
        main_metrics = compute_metrics(
            y_true=y_test,
            y_proba=y_test_proba,
            threshold=0.5,
        )

        main_row = {
            "model": "MLP_balanced_undersampling",
            "feature_set": "reduced",
            "test_shape": fold["test_shape"],
            "best_epoch": fold["best_epoch"],
            "best_val_loss": fold["best_val_loss"],
            "original_train_rows": fold["original_train_rows"],
            "balanced_train_rows": fold["balanced_train_rows"],
            "original_train_positive_rate": fold["original_train_positive_rate"],
            "balanced_train_positive_rate": fold["balanced_train_positive_rate"],
            "val_rows": fold["val_rows"],
            "val_positive_rate": fold["val_positive_rate"],
            "test_rows": fold["test_rows"],
            "test_positive_rate": fold["test_positive_rate"],
        }

        main_row.update(main_metrics)
        by_shape_rows.append(main_row)

        # Threshold sweep
        for threshold in THRESHOLDS:
            metrics = compute_metrics(
                y_true=y_test,
                y_proba=y_test_proba,
                threshold=threshold,
            )

            threshold_row = {
                "model": "MLP_balanced_undersampling",
                "feature_set": "reduced",
                "test_shape": fold["test_shape"],
                "best_epoch": fold["best_epoch"],
                "best_val_loss": fold["best_val_loss"],
                "test_rows": fold["test_rows"],
                "test_positive_rate": fold["test_positive_rate"],
            }

            threshold_row.update(metrics)
            threshold_rows.append(threshold_row)

            if threshold in [0.5, 0.6, 0.7]:
                y_pred = (y_test_proba >= threshold).astype(int)
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

                threshold_tag = str(threshold).replace(".", "p")
                cm_path = FIGURE_DIR / (
                    f"mlp_balanced_undersampling_confusion_matrix_"
                    f"{test_shape}_thr_{threshold_tag}.png"
                )

                plot_confusion_matrix(
                    cm,
                    (
                        "MLP balanced undersampling | "
                        f"test={test_shape} | threshold={threshold}"
                    ),
                    cm_path,
                )

    by_shape_df = pd.DataFrame(by_shape_rows)
    threshold_df = pd.DataFrame(threshold_rows)

    by_shape_path = TABLE_DIR / "mlp_balanced_undersampling_results_by_shape.csv"
    by_shape_df.to_csv(by_shape_path, index=False)

    avg_row = {
        "model": "MLP_balanced_undersampling",
        "feature_set": "reduced",
        "num_folds": len(by_shape_df),
        "accuracy_mean": by_shape_df["accuracy"].mean(),
        "accuracy_std": by_shape_df["accuracy"].std(),
        "precision_mean": by_shape_df["precision"].mean(),
        "precision_std": by_shape_df["precision"].std(),
        "recall_mean": by_shape_df["recall"].mean(),
        "recall_std": by_shape_df["recall"].std(),
        "f1_mean": by_shape_df["f1"].mean(),
        "f1_std": by_shape_df["f1"].std(),
        "roc_auc_mean": by_shape_df["roc_auc"].mean(),
        "roc_auc_std": by_shape_df["roc_auc"].std(),
        "acceptance_rate_mean": by_shape_df["acceptance_rate"].mean(),
        "acceptance_rate_std": by_shape_df["acceptance_rate"].std(),
    }

    avg_df = pd.DataFrame([avg_row])
    avg_path = TABLE_DIR / "mlp_balanced_undersampling_results_average.csv"
    avg_df.to_csv(avg_path, index=False)

    threshold_by_shape_path = (
        TABLE_DIR / "mlp_balanced_undersampling_threshold_results_by_shape.csv"
    )
    threshold_df.to_csv(threshold_by_shape_path, index=False)

    threshold_avg_df = summarize_results_by_threshold(threshold_df)
    threshold_avg_path = (
        TABLE_DIR / "mlp_balanced_undersampling_threshold_results_average.csv"
    )
    threshold_avg_df.to_csv(threshold_avg_path, index=False)

    print("\n" + "=" * 80)
    print("Balanced undersampling average result at threshold 0.5:")
    print(avg_df.to_string(index=False))

    print("\nThreshold average results:")
    print(threshold_avg_df.to_string(index=False))

    print("\nSaved:")
    print(f"  {by_shape_path}")
    print(f"  {avg_path}")
    print(f"  {threshold_by_shape_path}")
    print(f"  {threshold_avg_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
