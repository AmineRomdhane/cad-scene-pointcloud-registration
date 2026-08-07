#!/usr/bin/env python3
"""
MLP real-only v3 with:
- hard 0/1 labels
- ambiguous-label removal for training
- case-balanced training only
- NO forced class-balanced sampling
- fixed pos_weight = 1.0
- LR = 3e-4
- dropout = 0
- early stopping on validation ROC-AUC

Training labels:
    positive if label_distance < 0.08
    negative if label_distance > 0.14
    discard otherwise

Evaluation:
    1) original target_weight labels from the dataset
    2) clear margin labels only, using positive < 0.08 and negative > 0.14
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

TEST_NAME = "mlp_real_only_v3_margin_case_balanced_only_auc"
OUT_DIR = BASE_DIR / "results" / "by_test" / TEST_NAME

MODEL_NAME = "MLP_real_only_v3_margin_case_balanced_only_auc"

FEATURES = [
    "distance_T0",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
]

HARD_TARGET = "target_weight"
LABEL_DISTANCE = "label_distance"
MARGIN_TARGET = "margin_target"

POSITIVE_DISTANCE = 0.08
NEGATIVE_DISTANCE = 0.14

RANDOM_STATE = 42

LR = 3e-4
POS_WEIGHT_VALUE = 1.0
DROPOUT = 0.0

EPOCHS = 200
BATCH_SIZE = 512
WEIGHT_DECAY = 1e-4
PATIENCE = 20

VAL_CASE_FRACTION = 0.20
ROWS_PER_CASE_PER_EPOCH = 1000

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


class CorrespondenceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
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


def add_margin_labels(df):
    df = df.copy()
    df[MARGIN_TARGET] = np.nan

    df.loc[df[LABEL_DISTANCE] < POSITIVE_DISTANCE, MARGIN_TARGET] = 1
    df.loc[df[LABEL_DISTANCE] > NEGATIVE_DISTANCE, MARGIN_TARGET] = 0

    return df


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
        for start in range(0, len(X_tensor), 4096):
            xb = X_tensor[start:start + 4096]
            logits = model(xb)
            prob = torch.sigmoid(logits)
            scores.append(prob.cpu().numpy())

    return np.concatenate(scores, axis=0)


def sample_case_balanced_only(train_df, rows_per_case, seed):
    """
    For each real case:
    - sample the same number of rows
    - preserve the case's natural positive/negative ratio
    - use replacement if the case has fewer rows than rows_per_case
    """

    sampled_parts = []
    count_rows = []

    case_ids = sorted(train_df["real_case_id"].unique())

    for case_idx, case_id in enumerate(case_ids):
        case_df = train_df[train_df["real_case_id"] == case_id].copy()

        if len(case_df) == 0:
            continue

        replace = len(case_df) < rows_per_case

        sampled_case = case_df.sample(
            n=rows_per_case,
            replace=replace,
            random_state=seed + 1009 * case_idx,
        )

        sampled_parts.append(sampled_case)

        count_rows.append({
            "real_case_id": case_id,
            "original_rows_margin": len(case_df),
            "original_positive": int((case_df[MARGIN_TARGET] == 1).sum()),
            "original_negative": int((case_df[MARGIN_TARGET] == 0).sum()),
            "original_positive_rate": float(case_df[MARGIN_TARGET].mean()),
            "sampled_rows": len(sampled_case),
            "sampled_positive": int((sampled_case[MARGIN_TARGET] == 1).sum()),
            "sampled_negative": int((sampled_case[MARGIN_TARGET] == 0).sum()),
            "sampled_positive_rate": float(sampled_case[MARGIN_TARGET].mean()),
            "sampled_with_replacement": replace,
        })

    sampled_df = pd.concat(sampled_parts, axis=0, ignore_index=True)

    sampled_df = sampled_df.sample(
        frac=1.0,
        replace=False,
        random_state=seed + 99991,
    ).reset_index(drop=True)

    return sampled_df, pd.DataFrame(count_rows)


def evaluate_validation(model, val_df, scaler, criterion, device):
    X_val = scaler.transform(val_df[FEATURES].values)
    y_val = val_df[MARGIN_TARGET].values.astype(np.float32)

    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    model.eval()
    val_losses = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            val_losses.append(loss.item())

    score = predict_score(model, X_val, device)

    val_loss = float(np.mean(val_losses))
    val_auc = safe_auc(y_val.astype(int), score)

    return val_loss, val_auc


def train_one_fold(train_margin_df, val_margin_df, test_original_df, test_margin_df, heldout_case, device, fold_seed):
    scaler = StandardScaler()
    scaler.fit(train_margin_df[FEATURES].values)

    model = CorrespondenceMLP(input_dim=len(FEATURES)).to(device)

    pos_weight = torch.tensor([POS_WEIGHT_VALUE], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_auc = -np.inf
    best_val_loss = np.inf
    best_epoch = 0
    best_state = None
    patience_count = 0

    history_rows = []
    sampling_rows = []

    for epoch in range(1, EPOCHS + 1):
        epoch_seed = fold_seed + epoch * 10000

        sampled_train_df, sample_counts_df = sample_case_balanced_only(
            train_df=train_margin_df,
            rows_per_case=ROWS_PER_CASE_PER_EPOCH,
            seed=epoch_seed,
        )

        sample_counts_df["heldout_real_case"] = heldout_case
        sample_counts_df["epoch"] = epoch
        sampling_rows.extend(sample_counts_df.to_dict("records"))

        X_train_epoch = scaler.transform(sampled_train_df[FEATURES].values)
        y_train_epoch = sampled_train_df[MARGIN_TARGET].values.astype(np.float32)

        train_loader = make_loader(
            X_train_epoch,
            y_train_epoch,
            BATCH_SIZE,
            shuffle=True,
        )

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

        val_loss, val_auc = evaluate_validation(
            model=model,
            val_df=val_margin_df,
            scaler=scaler,
            criterion=criterion,
            device=device,
        )

        improved = False

        if not np.isnan(val_auc):
            if val_auc > best_val_auc + 1e-5:
                improved = True
        else:
            if val_loss < best_val_loss - 1e-6:
                improved = True

        if improved:
            best_val_auc = val_auc
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
            "val_auc_margin": val_auc,
            "best_epoch_so_far": best_epoch,
            "best_val_auc_so_far": best_val_auc,
            "best_val_loss_so_far": best_val_loss,
            "patience_count": patience_count,
            "sampled_train_rows": len(sampled_train_df),
            "sampled_train_positive_rate": float(sampled_train_df[MARGIN_TARGET].mean()),
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
                f"Best epoch: {best_epoch}, best val AUC: {best_val_auc:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_results = []
    threshold_rows = []

    eval_sets = [
        {
            "eval_set": "original_target_0p10",
            "df": test_original_df,
            "target_col": HARD_TARGET,
        },
        {
            "eval_set": "clear_margin_0p08_0p14",
            "df": test_margin_df,
            "target_col": MARGIN_TARGET,
        },
    ]

    for item in eval_sets:
        eval_set = item["eval_set"]
        eval_df = item["df"].copy()
        target_col = item["target_col"]

        if len(eval_df) == 0:
            continue

        X_eval = scaler.transform(eval_df[FEATURES].values)
        y_eval = eval_df[target_col].values.astype(int)
        score = predict_score(model, X_eval, device)

        metrics_05 = compute_metrics(
            y_true=y_eval,
            score=score,
            threshold=0.5,
        )

        result_row = {
            "model": MODEL_NAME,
            "eval_set": eval_set,
            "heldout_real_case": heldout_case,
            "lr": LR,
            "pos_weight": POS_WEIGHT_VALUE,
            "dropout": DROPOUT,
            "positive_distance": POSITIVE_DISTANCE,
            "negative_distance": NEGATIVE_DISTANCE,
            "rows_per_case_per_epoch": ROWS_PER_CASE_PER_EPOCH,
            "best_epoch": best_epoch,
            "best_val_auc_margin": best_val_auc,
            "best_val_loss": best_val_loss,
            "train_margin_rows": len(train_margin_df),
            "val_margin_rows": len(val_margin_df),
            "test_rows": len(eval_df),
            "train_margin_positive_rate": float(train_margin_df[MARGIN_TARGET].mean()),
            "val_margin_positive_rate": float(val_margin_df[MARGIN_TARGET].mean()),
            "test_positive_rate": float(eval_df[target_col].mean()),
        }

        result_row.update(metrics_05)
        eval_results.append(result_row)

        for threshold in THRESHOLDS:
            m = compute_metrics(
                y_true=y_eval,
                score=score,
                threshold=threshold,
            )

            threshold_rows.append({
                "model": MODEL_NAME,
                "eval_set": eval_set,
                "heldout_real_case": heldout_case,
                "lr": LR,
                "pos_weight": POS_WEIGHT_VALUE,
                "dropout": DROPOUT,
                "positive_distance": POSITIVE_DISTANCE,
                "negative_distance": NEGATIVE_DISTANCE,
                **m,
            })

    return eval_results, threshold_rows, history_rows, sampling_rows


def summarize_average(df):
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
        "best_epoch",
        "best_val_auc_margin",
    ]

    rows = []

    for eval_set, group in df.groupby("eval_set"):
        row = {
            "model": MODEL_NAME,
            "eval_set": eval_set,
            "num_folds": group["heldout_real_case"].nunique(),
            "lr": LR,
            "pos_weight": POS_WEIGHT_VALUE,
            "dropout": DROPOUT,
            "positive_distance": POSITIVE_DISTANCE,
            "negative_distance": NEGATIVE_DISTANCE,
            "rows_per_case_per_epoch": ROWS_PER_CASE_PER_EPOCH,
        }

        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std()

        rows.append(row)

    return pd.DataFrame(rows)


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

    for (eval_set, threshold), group in df.groupby(["eval_set", "threshold"]):
        row = {
            "model": MODEL_NAME,
            "eval_set": eval_set,
            "num_folds": group["heldout_real_case"].nunique(),
            "threshold": threshold,
            "lr": LR,
            "pos_weight": POS_WEIGHT_VALUE,
            "dropout": DROPOUT,
            "positive_distance": POSITIVE_DISTANCE,
            "negative_distance": NEGATIVE_DISTANCE,
        }

        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std()

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["eval_set", "threshold"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(MODEL_NAME)
    print("=" * 100)
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Features: {FEATURES}")
    print(f"LR={LR}")
    print(f"pos_weight={POS_WEIGHT_VALUE}")
    print(f"dropout={DROPOUT}")
    print(f"positive if label_distance < {POSITIVE_DISTANCE}")
    print(f"negative if label_distance > {NEGATIVE_DISTANCE}")
    print(f"case-balanced only rows per case per epoch = {ROWS_PER_CASE_PER_EPOCH}")
    print("NO forced class balance")
    print("early stopping metric = validation ROC-AUC on margin labels")
    print("=" * 100)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["sample_id", HARD_TARGET, LABEL_DISTANCE] + FEATURES
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols).copy()

    df["sample_id"] = df["sample_id"].astype(str)
    df = df[df["sample_id"].str.startswith("real_")].copy()

    df["real_case_id"] = df["sample_id"].apply(get_real_case_id)
    df[HARD_TARGET] = df[HARD_TARGET].astype(int)

    df = add_margin_labels(df)

    margin_df = df.dropna(subset=[MARGIN_TARGET]).copy()
    margin_df[MARGIN_TARGET] = margin_df[MARGIN_TARGET].astype(int)

    real_case_ids = sorted(df["real_case_id"].unique())

    print(f"Real rows original:     {len(df)}")
    print(f"Real rows after margin: {len(margin_df)}")
    print(f"Discarded ambiguous:    {len(df) - len(margin_df)}")
    print(f"Retained ratio:         {len(margin_df) / max(len(df), 1):.4f}")
    print(f"Original pos rate:      {df[HARD_TARGET].mean():.4f}")
    print(f"Margin pos rate:        {margin_df[MARGIN_TARGET].mean():.4f}")
    print(f"Real cases:             {len(real_case_ids)}")
    print("-" * 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("-" * 100)

    result_rows = []
    threshold_rows_all = []
    history_rows_all = []
    sampling_rows_all = []
    split_rows = []

    for heldout_index, heldout_case in enumerate(real_case_ids):
        print("\n" + "=" * 100)
        print(f"Held-out real case: {heldout_case}")
        print("=" * 100)

        test_original_df = df[df["real_case_id"] == heldout_case].copy()
        test_margin_df = margin_df[margin_df["real_case_id"] == heldout_case].copy()

        trainval_margin_df = margin_df[margin_df["real_case_id"] != heldout_case].copy()

        available_cases = sorted(trainval_margin_df["real_case_id"].unique())

        num_val_cases = max(1, int(round(VAL_CASE_FRACTION * len(available_cases))))

        rng = np.random.default_rng(RANDOM_STATE + heldout_index)

        val_cases = set(
            rng.choice(
                available_cases,
                size=num_val_cases,
                replace=False,
            ).tolist()
        )

        val_margin_df = trainval_margin_df[trainval_margin_df["real_case_id"].isin(val_cases)].copy()
        train_margin_df = trainval_margin_df[~trainval_margin_df["real_case_id"].isin(val_cases)].copy()

        print(f"Train margin rows:       {len(train_margin_df)}")
        print(f"Val margin rows:         {len(val_margin_df)}")
        print(f"Test original rows:      {len(test_original_df)}")
        print(f"Test clear margin rows:  {len(test_margin_df)}")
        print(f"Train margin pos:        {train_margin_df[MARGIN_TARGET].mean():.4f}")
        print(f"Val margin pos:          {val_margin_df[MARGIN_TARGET].mean():.4f}")
        print(f"Test original pos:       {test_original_df[HARD_TARGET].mean():.4f}")
        print(f"Test clear margin pos:   {test_margin_df[MARGIN_TARGET].mean():.4f}")
        print(f"Validation cases:        {sorted(val_cases)}")

        split_rows.append({
            "heldout_real_case": heldout_case,
            "validation_cases": "|".join(sorted(val_cases)),
            "train_margin_rows": len(train_margin_df),
            "val_margin_rows": len(val_margin_df),
            "test_original_rows": len(test_original_df),
            "test_clear_margin_rows": len(test_margin_df),
            "train_margin_positive_rate": float(train_margin_df[MARGIN_TARGET].mean()),
            "val_margin_positive_rate": float(val_margin_df[MARGIN_TARGET].mean()),
            "test_original_positive_rate": float(test_original_df[HARD_TARGET].mean()),
            "test_clear_margin_positive_rate": float(test_margin_df[MARGIN_TARGET].mean()),
        })

        fold_seed = RANDOM_STATE + 100000 * heldout_index

        eval_results, threshold_rows, history_rows, sampling_rows = train_one_fold(
            train_margin_df=train_margin_df,
            val_margin_df=val_margin_df,
            test_original_df=test_original_df,
            test_margin_df=test_margin_df,
            heldout_case=heldout_case,
            device=device,
            fold_seed=fold_seed,
        )

        result_rows.extend(eval_results)
        threshold_rows_all.extend(threshold_rows)
        history_rows_all.extend(history_rows)
        sampling_rows_all.extend(sampling_rows)

        for row in eval_results:
            print(
                f"Result [{row['eval_set']}] | "
                f"AUC={row['roc_auc']:.3f} | "
                f"F1@0.5={row['f1']:.3f} | "
                f"precision={row['precision']:.3f} | "
                f"recall={row['recall']:.3f} | "
                f"acceptance={row['acceptance_rate']:.3f} | "
                f"best_epoch={row['best_epoch']} | "
                f"best_val_auc={row['best_val_auc_margin']:.3f}"
            )

    results_df = pd.DataFrame(result_rows)
    threshold_df = pd.DataFrame(threshold_rows_all)
    history_df = pd.DataFrame(history_rows_all)
    sampling_df = pd.DataFrame(sampling_rows_all)
    split_df = pd.DataFrame(split_rows)

    avg_df = summarize_average(results_df)
    threshold_avg_df = summarize_thresholds(threshold_df)

    results_path = OUT_DIR / f"{TEST_NAME}_results_by_case.csv"
    avg_path = OUT_DIR / f"{TEST_NAME}_results_average.csv"
    threshold_path = OUT_DIR / f"{TEST_NAME}_threshold_results_by_case.csv"
    threshold_avg_path = OUT_DIR / f"{TEST_NAME}_threshold_results_average.csv"
    history_path = OUT_DIR / f"{TEST_NAME}_training_history.csv"
    sampling_path = OUT_DIR / f"{TEST_NAME}_sampling_counts.csv"
    split_path = OUT_DIR / f"{TEST_NAME}_splits.csv"

    results_df.to_csv(results_path, index=False)
    avg_df.to_csv(avg_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    threshold_avg_df.to_csv(threshold_avg_path, index=False)
    history_df.to_csv(history_path, index=False)
    sampling_df.to_csv(sampling_path, index=False)
    split_df.to_csv(split_path, index=False)

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
    print(f"  {sampling_path}")
    print(f"  {split_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
