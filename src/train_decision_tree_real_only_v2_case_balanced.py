#!/usr/bin/env python3
"""
Decision tree diagnostic on real-only v2 data with case-balanced training samples.

Goal:
- Use only real correspondence samples from v2.
- Evaluate with leave-one-real-case-out.
- During training, sample the same number of rows from each real case.
- Keep the held-out test case unchanged.
- Compare feature importance after removing the domination of large cases.

Input:
- results/learning_data_synthetic_plus_real_curated_clean_v2/all_correspondences.csv

Outputs:
- results/tables/decision_tree_real_only_v2_case_balanced_results_by_run.csv
- results/tables/decision_tree_real_only_v2_case_balanced_results_by_case.csv
- results/tables/decision_tree_real_only_v2_case_balanced_results_average.csv
- results/tables/decision_tree_real_only_v2_case_balanced_feature_importance_by_run.csv
- results/tables/decision_tree_real_only_v2_case_balanced_feature_importance_average.csv
- results/tables/decision_tree_real_only_v2_case_balanced_permutation_importance_by_run.csv
- results/tables/decision_tree_real_only_v2_case_balanced_permutation_importance_average.csv
- results/tables/decision_tree_real_only_v2_case_balanced_train_sampling_counts.csv
- results/tables/decision_tree_real_only_v2_case_balanced_all_real_feature_importance.csv
- results/tables/decision_tree_real_only_v2_case_balanced_rules_all_real.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    BASE_DIR
    / "results"
    / "learning_data_synthetic_plus_real_curated_clean_v2"
    / "all_correspondences.csv"
)

TABLE_DIR = BASE_DIR / "results" / "tables"

TARGET = "target_weight"

FEATURES = [
    "distance_T0",
    "point_to_plane_residual",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
]

MODEL_NAME = "DecisionTree_real_only_v2_case_balanced"

RANDOM_STATE = 42

MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 50
THRESHOLD = 0.5

# Same number of training correspondences per real case.
# If a case has fewer rows than this, it is sampled with replacement.
ROWS_PER_CASE = 1000

# Repeating reduces randomness caused by sampling.
N_BALANCE_REPEATS = 5

N_PERMUTATION_REPEATS = 5


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


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, y_score)


def compute_metrics(y_true, y_proba, threshold=0.5):
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


def make_tree(random_state):
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=random_state,
    )


def sample_case_balanced(train_df, rows_per_case, seed):
    sampled_parts = []
    count_rows = []

    case_ids = sorted(train_df["real_case_id"].unique())

    for idx, case_id in enumerate(case_ids):
        case_df = train_df[train_df["real_case_id"] == case_id].copy()

        replace = len(case_df) < rows_per_case

        sampled = case_df.sample(
            n=rows_per_case,
            replace=replace,
            random_state=seed + 1009 * idx,
        )

        sampled_parts.append(sampled)

        count_rows.append({
            "real_case_id": case_id,
            "original_rows": len(case_df),
            "sampled_rows": len(sampled),
            "sampled_with_replacement": replace,
            "original_positive_rate": float(case_df[TARGET].mean()),
            "sampled_positive_rate": float(sampled[TARGET].mean()),
        })

    sampled_df = pd.concat(sampled_parts, axis=0, ignore_index=True)

    # Shuffle final training rows.
    sampled_df = sampled_df.sample(
        frac=1.0,
        replace=False,
        random_state=seed + 99991,
    ).reset_index(drop=True)

    counts_df = pd.DataFrame(count_rows)

    return sampled_df, counts_df


def summarize_average(results_df):
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
        "num_folds": results_df["heldout_real_case"].nunique(),
        "num_runs": len(results_df),
        "n_balance_repeats": N_BALANCE_REPEATS,
        "rows_per_case": ROWS_PER_CASE,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
    }

    for col in metric_cols:
        row[f"{col}_mean"] = results_df[col].mean()
        row[f"{col}_std"] = results_df[col].std()

    return pd.DataFrame([row])


def summarize_by_case(results_df):
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

    agg_dict = {
        "test_rows": "first",
        "test_positive_rate": "first",
        "train_balanced_rows": "mean",
        "train_balanced_positive_rate": "mean",
    }

    for col in metric_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, "std")

    rows = []

    for case_id, group in results_df.groupby("heldout_real_case"):
        row = {
            "model": MODEL_NAME,
            "heldout_real_case": case_id,
            "num_repeats": len(group),
            "test_rows": int(group["test_rows"].iloc[0]),
            "test_positive_rate": float(group["test_positive_rate"].iloc[0]),
            "train_balanced_rows_mean": float(group["train_balanced_rows"].mean()),
            "train_balanced_positive_rate_mean": float(group["train_balanced_positive_rate"].mean()),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std())

        rows.append(row)

    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Decision tree real-only v2 with case-balanced training")
    print("=" * 100)
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Features: {FEATURES}")
    print(f"ROWS_PER_CASE={ROWS_PER_CASE}")
    print(f"N_BALANCE_REPEATS={N_BALANCE_REPEATS}")
    print(f"max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF}")
    print("=" * 100)

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["sample_id", TARGET] + FEATURES
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols).copy()

    df["sample_id"] = df["sample_id"].astype(str)
    df = df[df["sample_id"].str.startswith("real_")].copy()

    df["real_case_id"] = df["sample_id"].apply(get_real_case_id)
    df[TARGET] = df[TARGET].astype(int)

    real_case_ids = sorted(df["real_case_id"].unique())

    if not real_case_ids:
        raise RuntimeError("No real cases found.")

    print(f"Real-only rows: {len(df)}")
    print(f"Real cases:     {len(real_case_ids)}")
    print(f"Positive rate:  {df[TARGET].mean():.4f}")
    print("-" * 100)

    result_rows = []
    importance_rows = []
    permutation_rows = []
    sampling_count_rows = []

    for heldout_case in real_case_ids:
        test_df = df[df["real_case_id"] == heldout_case].copy()
        train_df = df[df["real_case_id"] != heldout_case].copy()

        X_test = test_df[FEATURES].values
        y_test = test_df[TARGET].values

        for repeat_idx in range(N_BALANCE_REPEATS):
            seed = RANDOM_STATE + 100000 * repeat_idx + 17 * len(heldout_case)

            train_balanced_df, counts_df = sample_case_balanced(
                train_df=train_df,
                rows_per_case=ROWS_PER_CASE,
                seed=seed,
            )

            for _, count_row in counts_df.iterrows():
                sampling_count_rows.append({
                    "heldout_real_case": heldout_case,
                    "repeat_idx": repeat_idx,
                    **count_row.to_dict(),
                })

            X_train = train_balanced_df[FEATURES].values
            y_train = train_balanced_df[TARGET].values

            clf = make_tree(random_state=seed)
            clf.fit(X_train, y_train)

            y_proba = clf.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, y_proba, threshold=THRESHOLD)

            row = {
                "model": MODEL_NAME,
                "heldout_real_case": heldout_case,
                "repeat_idx": repeat_idx,
                "train_original_rows": len(train_df),
                "train_balanced_rows": len(train_balanced_df),
                "test_rows": len(test_df),
                "train_original_positive_rate": float(train_df[TARGET].mean()),
                "train_balanced_positive_rate": float(np.mean(y_train)),
                "test_positive_rate": float(np.mean(y_test)),
                "rows_per_case": ROWS_PER_CASE,
                "max_depth": MAX_DEPTH,
                "min_samples_leaf": MIN_SAMPLES_LEAF,
            }

            row.update(metrics)
            result_rows.append(row)

            for feature, importance in zip(FEATURES, clf.feature_importances_):
                importance_rows.append({
                    "heldout_real_case": heldout_case,
                    "repeat_idx": repeat_idx,
                    "feature": feature,
                    "gini_importance": float(importance),
                })

            if len(np.unique(y_test)) >= 2:
                perm = permutation_importance(
                    clf,
                    X_test,
                    y_test,
                    scoring="roc_auc",
                    n_repeats=N_PERMUTATION_REPEATS,
                    random_state=seed + 555,
                    n_jobs=-1,
                )

                for i, feature in enumerate(FEATURES):
                    permutation_rows.append({
                        "heldout_real_case": heldout_case,
                        "repeat_idx": repeat_idx,
                        "feature": feature,
                        "permutation_importance_mean": float(perm.importances_mean[i]),
                        "permutation_importance_std": float(perm.importances_std[i]),
                    })

            print(
                f"{heldout_case} | repeat={repeat_idx} | "
                f"AUC={metrics['roc_auc']:.3f} | "
                f"F1={metrics['f1']:.3f} | "
                f"precision={metrics['precision']:.3f} | "
                f"recall={metrics['recall']:.3f} | "
                f"acceptance={metrics['acceptance_rate']:.3f}"
            )

    results_df = pd.DataFrame(result_rows)
    importance_df = pd.DataFrame(importance_rows)
    permutation_df = pd.DataFrame(permutation_rows)
    sampling_counts_df = pd.DataFrame(sampling_count_rows)

    avg_df = summarize_average(results_df)
    by_case_df = summarize_by_case(results_df)

    importance_avg_df = (
        importance_df
        .groupby("feature", as_index=False)
        .agg(
            gini_importance_mean=("gini_importance", "mean"),
            gini_importance_std=("gini_importance", "std"),
        )
        .sort_values("gini_importance_mean", ascending=False)
    )

    if not permutation_df.empty:
        permutation_avg_df = (
            permutation_df
            .groupby("feature", as_index=False)
            .agg(
                permutation_importance_mean=("permutation_importance_mean", "mean"),
                permutation_importance_std=("permutation_importance_mean", "std"),
            )
            .sort_values("permutation_importance_mean", ascending=False)
        )
    else:
        permutation_avg_df = pd.DataFrame(columns=[
            "feature",
            "permutation_importance_mean",
            "permutation_importance_std",
        ])

    # Final interpretable tree trained on all real cases using one case-balanced sample.
    final_balanced_df, final_counts_df = sample_case_balanced(
        train_df=df,
        rows_per_case=ROWS_PER_CASE,
        seed=RANDOM_STATE,
    )

    final_tree = make_tree(random_state=RANDOM_STATE)
    final_tree.fit(final_balanced_df[FEATURES].values, final_balanced_df[TARGET].values)

    final_importance_df = pd.DataFrame({
        "feature": FEATURES,
        "gini_importance": final_tree.feature_importances_,
    }).sort_values("gini_importance", ascending=False)

    rules = export_text(
        final_tree,
        feature_names=FEATURES,
        decimals=5,
        spacing=3,
    )

    results_path = TABLE_DIR / "decision_tree_real_only_v2_case_balanced_results_by_run.csv"
    by_case_path = TABLE_DIR / "decision_tree_real_only_v2_case_balanced_results_by_case.csv"
    avg_path = TABLE_DIR / "decision_tree_real_only_v2_case_balanced_results_average.csv"

    importance_path = TABLE_DIR / "decision_tree_real_only_v2_case_balanced_feature_importance_by_run.csv"
    importance_avg_path = TABLE_DIR / "decision_tree_real_only_v2_case_balanced_feature_importance_average.csv"

    permutation_path = TABLE_DIR / "decision_tree_real_only_v2_case_balanced_permutation_importance_by_run.csv"
    permutation_avg_path = TABLE_DIR / "decision_tree_real_only_v2_case_balanced_permutation_importance_average.csv"

    sampling_counts_path = TABLE_DIR / "decision_tree_real_only_v2_case_balanced_train_sampling_counts.csv"

    final_importance_path = TABLE_DIR / "decision_tree_real_only_v2_case_balanced_all_real_feature_importance.csv"
    rules_path = TABLE_DIR / "decision_tree_real_only_v2_case_balanced_rules_all_real.txt"

    results_df.to_csv(results_path, index=False)
    by_case_df.to_csv(by_case_path, index=False)
    avg_df.to_csv(avg_path, index=False)

    importance_df.to_csv(importance_path, index=False)
    importance_avg_df.to_csv(importance_avg_path, index=False)

    permutation_df.to_csv(permutation_path, index=False)
    permutation_avg_df.to_csv(permutation_avg_path, index=False)

    sampling_counts_df.to_csv(sampling_counts_path, index=False)

    final_importance_df.to_csv(final_importance_path, index=False)
    rules_path.write_text(rules)

    print("\n" + "=" * 100)
    print("Average metrics:")
    print(avg_df.to_string(index=False))

    print("\nAverage metrics by held-out real case:")
    print(by_case_df.to_string(index=False))

    print("\nAverage Gini feature importance:")
    print(importance_avg_df.to_string(index=False))

    print("\nAverage permutation importance on held-out real cases:")
    print(permutation_avg_df.to_string(index=False))

    print("\nFinal tree trained on all real data with case-balanced sampling:")
    print(final_importance_df.to_string(index=False))

    print("\nSaved:")
    print(f"  {results_path}")
    print(f"  {by_case_path}")
    print(f"  {avg_path}")
    print(f"  {importance_path}")
    print(f"  {importance_avg_path}")
    print(f"  {permutation_path}")
    print(f"  {permutation_avg_path}")
    print(f"  {sampling_counts_path}")
    print(f"  {final_importance_path}")
    print(f"  {rules_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
