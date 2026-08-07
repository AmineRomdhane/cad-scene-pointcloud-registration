from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path("/home/roam5170/GMC6003_registration")
OUT_DIR = BASE_DIR / "results/by_test/confusion_matrices_for_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def has_cols(df, cols):
    return all(c in df.columns for c in cols)


def find_candidate_csvs():
    csvs = list((BASE_DIR / "results").rglob("*.csv"))
    valid = []

    for p in csvs:
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue

        cols = set(df.columns)

        has_threshold = "threshold" in cols
        has_counts = (
            {"tn", "fp", "fn", "tp"}.issubset(cols)
            or {"tn_sum", "fp_sum", "fn_sum", "tp_sum"}.issubset(cols)
            or {"tn_mean", "fp_mean", "fn_mean", "tp_mean"}.issubset(cols)
        )

        if has_threshold and has_counts:
            valid.append(p)

    return valid


def print_candidates(candidates):
    print("\nCandidate CSV files with threshold and confusion counts:")
    for i, p in enumerate(candidates):
        print(f"[{i}] {p}")


def choose_csv(candidates, keywords, avoid_average=True):
    scored = []

    for p in candidates:
        s = str(p).lower()
        score = 0

        for kw in keywords:
            if kw.lower() in s:
                score += 10

        if avoid_average and "average" in s:
            score -= 20

        if "threshold" in s:
            score += 3

        scored.append((score, p))

    scored.sort(reverse=True, key=lambda x: x[0])

    if not scored or scored[0][0] <= 0:
        raise FileNotFoundError(
            f"No suitable CSV found for keywords={keywords}. "
            "Check the printed candidate list."
        )

    return scored[0][1]


def filter_rows(df, threshold=0.5, model_name=None, feature_set=None):
    mask = np.ones(len(df), dtype=bool)

    if "threshold" in df.columns:
        mask &= np.isclose(df["threshold"].astype(float), float(threshold))

    if model_name is not None and "model" in df.columns:
        model_mask = df["model"].astype(str) == str(model_name)
        if model_mask.any():
            mask &= model_mask
        else:
            print(f"Warning: model '{model_name}' not found. Using threshold filter only.")

    if feature_set is not None and "feature_set" in df.columns:
        fs_mask = df["feature_set"].astype(str) == str(feature_set)
        if fs_mask.any():
            mask &= fs_mask
        else:
            print(f"Warning: feature_set '{feature_set}' not found. Using other filters only.")

    selected = df.loc[mask].copy()

    if len(selected) == 0:
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nFirst rows:")
        print(df.head())
        raise ValueError("No rows selected. Check threshold/model/feature_set.")

    return selected


def confusion_from_rows(rows):
    """
    Returns confusion matrix:
        [[TN, FP],
         [FN, TP]]

    Priority:
    1. exact counts: tn/fp/fn/tp summed over selected rows
    2. sum columns: tn_sum/fp_sum/fn_sum/tp_sum
    3. average columns: tn_mean/fp_mean/fn_mean/tp_mean, only if no exact data exists
    """

    if has_cols(rows, ["tn", "fp", "fn", "tp"]):
        source_type = "summed exact counts from detailed rows"
        tn = rows["tn"].sum()
        fp = rows["fp"].sum()
        fn = rows["fn"].sum()
        tp = rows["tp"].sum()

    elif has_cols(rows, ["tn_sum", "fp_sum", "fn_sum", "tp_sum"]):
        source_type = "summed counts from *_sum columns"
        tn = rows["tn_sum"].sum()
        fp = rows["fp_sum"].sum()
        fn = rows["fn_sum"].sum()
        tp = rows["tp_sum"].sum()

    elif has_cols(rows, ["tn_mean", "fp_mean", "fn_mean", "tp_mean"]):
        source_type = "AVERAGE counts from *_mean columns"
        if len(rows) > 1:
            print("Warning: multiple average rows selected. Using the first one.")
        r = rows.iloc[0]
        tn = r["tn_mean"]
        fp = r["fp_mean"]
        fn = r["fn_mean"]
        tp = r["tp_mean"]

    else:
        raise KeyError("No usable TN/FP/FN/TP columns found.")

    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    return cm, source_type


def plot_confusion_matrix_white(cm, out_path):
    labels = ["Unreliable", "Reliable"]

    fig, ax = plt.subplots(figsize=(5.0, 4.4))

    # White cells with black borders: IEEE-friendly and readable.
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                facecolor="white",
                edgecolor="black",
                linewidth=1.2,
            )
            ax.add_patch(rect)

            value = int(round(cm[i, j]))
            ax.text(
                j,
                i,
                f"{value}",
                ha="center",
                va="center",
                fontsize=16,
                color="black",
            )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_csv(cm, out_csv):
    df = pd.DataFrame(
        cm,
        index=["true_unreliable", "true_reliable"],
        columns=["pred_unreliable", "pred_reliable"],
    )
    df.to_csv(out_csv)


def process_matrix(name, csv_path, threshold, model_name, feature_set, png_name, csv_name):
    print(f"\n=== {name} ===")
    print(f"Using CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    rows = filter_rows(
        df,
        threshold=threshold,
        model_name=model_name,
        feature_set=feature_set,
    )

    cm, source_type = confusion_from_rows(rows)

    print(f"Source type: {source_type}")
    print("Confusion matrix:")
    print("                 pred_unreliable   pred_reliable")
    print(f"true_unreliable {cm[0,0]:16.0f} {cm[0,1]:15.0f}")
    print(f"true_reliable   {cm[1,0]:16.0f} {cm[1,1]:15.0f}")

    if "AVERAGE" in source_type:
        print(
            "Warning: this is an average-count matrix, not exact pooled counts. "
            "It is usable as an average summary, but not as exact total counts."
        )

    out_png = OUT_DIR / png_name
    out_csv = OUT_DIR / csv_name

    plot_confusion_matrix_white(cm, out_png)
    save_csv(cm, out_csv)

    print(f"Saved PNG: {out_png}")
    print(f"Saved CSV: {out_csv}")


def main():
    candidates = find_candidate_csvs()
    print_candidates(candidates)

    # Synthetic: prefer a detailed threshold CSV, not an average CSV.
    synthetic_csv = choose_csv(
        candidates,
        keywords=["mlp", "threshold"],
        avoid_average=True,
    )

    # Real v3: prefer clean_v3 / holdout threshold detailed CSV.
    real_csv = choose_csv(
        candidates,
        keywords=["real", "v3", "threshold"],
        avoid_average=True,
    )

    process_matrix(
        name="Synthetic evaluation",
        csv_path=synthetic_csv,
        threshold=0.5,
        model_name="MLP_weighted_reduced",
        feature_set=None,
        png_name="synthetic_confusion_matrix_threshold_0p5.png",
        csv_name="synthetic_confusion_matrix_threshold_0p5.csv",
    )

    process_matrix(
        name="Real v3 evaluation",
        csv_path=real_csv,
        threshold=0.5,
        model_name="MLP_real_holdout_clean_v3",
        feature_set="reduced",
        png_name="real_v3_confusion_matrix_threshold_0p5.png",
        csv_name="real_v3_confusion_matrix_threshold_0p5.csv",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
