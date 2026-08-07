from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path("/home/roam5170/GMC6003_registration")

RESULTS_BY_SHAPE = BASE_DIR / "results/by_test/mlp_synthetic_plus_real_clean/mlp_synthetic_plus_real_clean_results_by_shape.csv"

OUT_DIR = BASE_DIR / "results/by_test/mlp_synthetic_plus_real_clean_summary"
CM_DIR = OUT_DIR / "confusion_matrices"
LOSS_DIR = OUT_DIR / "loss_curves"

CM_DIR.mkdir(parents=True, exist_ok=True)
LOSS_DIR.mkdir(parents=True, exist_ok=True)


def safe_slug(text):
    text = str(text).strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", text)


def reconstruct_cm_from_metrics(row):
    """
    Reconstruct confusion matrix from:
    test_rows, test_positive_rate, test_precision, test_recall.

    Matrix layout:
        [[TN, FP],
         [FN, TP]]
    """

    total = int(round(row["test_rows"]))
    positive = int(round(row["test_rows"] * row["test_positive_rate"]))
    negative = total - positive

    recall = float(row["test_recall"])
    precision = float(row["test_precision"])

    tp = int(round(recall * positive))
    fn = positive - tp

    if precision > 0:
        predicted_positive = int(round(tp / precision))
    else:
        predicted_positive = 0

    fp = predicted_positive - tp
    fp = max(0, min(fp, negative))

    tn = negative - fp

    return np.array([[tn, fp], [fn, tp]], dtype=int)


def compute_metrics_from_cm(cm):
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    accuracy = (tp + tn) / max(cm.sum(), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return accuracy, precision, recall, f1


def plot_confusion_matrix(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#e9e9e9")

    im = ax.imshow(cm, cmap="viridis")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Bad", "Reliable"], fontsize=18)
    ax.set_yticklabels(["Bad", "Reliable"], fontsize=18)

    ax.set_xlabel("Predicted label", fontsize=18)
    ax.set_ylabel("True label", fontsize=18)
    ax.set_title(title, fontsize=22, pad=14)

    vmax = cm.max()

    for i in range(2):
        for j in range(2):
            value = int(cm[i, j])
            color = "white" if cm[i, j] > 0.55 * vmax else "black"
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=20,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(history_csv, save_path):
    df = pd.read_csv(history_csv)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#e9e9e9")

    ax.plot(df["epoch"], df["train_loss"], linewidth=2.8, label="Train loss")
    ax.plot(df["epoch"], df["val_loss"], linewidth=2.8, label="Validation loss")

    title = history_csv.stem.replace("mlp_synthetic_plus_real_clean_training_history_", "")
    title = title.replace("_", " ")

    ax.set_title(f"MLP loss curve ({title})", fontsize=22, pad=14)
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel("BCE loss", fontsize=18)

    ax.grid(True, alpha=0.8, linewidth=1.5)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    if not RESULTS_BY_SHAPE.exists():
        raise FileNotFoundError(f"Missing file: {RESULTS_BY_SHAPE}")

    df = pd.read_csv(RESULTS_BY_SHAPE)

    print("Using:")
    print(RESULTS_BY_SHAPE)
    print()

    summary_rows = []

    # ------------------------------------------------------------------
    # 1) Individual confusion matrices by feature set and held-out shape
    # ------------------------------------------------------------------
    for _, row in df.iterrows():
        model = str(row["model"])
        feature_set = str(row["feature_set"])
        test_shape = str(row["test_shape"])

        cm = reconstruct_cm_from_metrics(row)
        acc, prec, rec, f1 = compute_metrics_from_cm(cm)

        title = f"MLP confusion matrix ({feature_set}, test={test_shape})"
        out_png = CM_DIR / f"mlp_confusion_matrix_{safe_slug(feature_set)}_{safe_slug(test_shape)}.png"

        plot_confusion_matrix(cm, title, out_png)

        summary_rows.append({
            "model": model,
            "feature_set": feature_set,
            "test_shape": test_shape,
            "tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1],
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "figure": str(out_png),
        })

        print(f"[OK] {out_png}")

    summary = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "confusion_matrix_summary_by_shape.csv"
    summary.to_csv(summary_csv, index=False)

    print()
    print(f"[OK] saved summary: {summary_csv}")

    # ------------------------------------------------------------------
    # 2) Pooled confusion matrices by feature set
    # ------------------------------------------------------------------
    pooled_rows = []

    for feature_set, group in summary.groupby("feature_set"):
        cm = np.array([
            [group["tn"].sum(), group["fp"].sum()],
            [group["fn"].sum(), group["tp"].sum()],
        ], dtype=int)

        acc, prec, rec, f1 = compute_metrics_from_cm(cm)

        title = f"MLP pooled confusion matrix ({feature_set})"
        out_png = CM_DIR / f"mlp_confusion_matrix_pooled_{safe_slug(feature_set)}.png"

        plot_confusion_matrix(cm, title, out_png)

        pooled_rows.append({
            "feature_set": feature_set,
            "tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1],
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "figure": str(out_png),
        })

        print(f"[OK] pooled {feature_set}: {out_png}")
        print(f"     TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
        print(f"     acc={acc:.3f} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")

    pooled = pd.DataFrame(pooled_rows)
    pooled_csv = OUT_DIR / "confusion_matrix_summary_pooled_by_feature_set.csv"
    pooled.to_csv(pooled_csv, index=False)

    print()
    print(f"[OK] saved pooled summary: {pooled_csv}")

    # ------------------------------------------------------------------
    # 3) Loss curves
    # ------------------------------------------------------------------
    history_files = sorted(
        (BASE_DIR / "results/by_test/mlp_synthetic_plus_real_clean").glob(
            "mlp_synthetic_plus_real_clean_training_history_*.csv"
        )
    )

    if len(history_files) == 0:
        print("[WARN] no training history files found.")
    else:
        for hist_csv in history_files:
            name = hist_csv.stem.replace("mlp_synthetic_plus_real_clean_training_history_", "")
            out_png = LOSS_DIR / f"mlp_loss_curve_{safe_slug(name)}.png"
            plot_loss_curve(hist_csv, out_png)
            print(f"[OK] loss curve: {out_png}")

    print()
    print("Done.")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
