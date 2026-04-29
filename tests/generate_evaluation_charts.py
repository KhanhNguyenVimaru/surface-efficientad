import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.lightning_model import EfficientAdModelSize


def to_scalar(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    if isinstance(value, (list, tuple)) and value:
        try:
            return float(value[0])
        except Exception:
            return None
    return None


def squeeze_batch(pred):
    while isinstance(pred, (list, tuple)) and len(pred) == 1:
        pred = pred[0]
    if isinstance(pred, dict):
        items = pred.items()
    elif hasattr(pred, "__dict__"):
        items = vars(pred).items()
    else:
        return {}
    out = {}
    for k, v in items:
        if hasattr(v, "detach"):
            v = v.detach().cpu()
        if hasattr(v, "numpy"):
            v = v.numpy()
        if isinstance(v, np.ndarray) and v.shape and v.shape[0] == 1:
            v = v.squeeze(0)
        elif isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        out[k] = v
    return out


def _get_attr(obj, attr, default=None):
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def get_test_dataset(datamodule):
    if hasattr(datamodule, "test_data") and datamodule.test_data is not None:
        return datamodule.test_data
    dl = datamodule.test_dataloader()
    if isinstance(dl, list):
        dl = dl[0]
    return dl.dataset


def collect_predictions(engine, model, datamodule, ckpt_path):
    print("Collecting predictions for evaluation charts...")
    raw_predictions = engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    if not raw_predictions:
        raise ValueError("No predictions returned.")
    predictions = [squeeze_batch(p) for p in raw_predictions]
    test_dataset = get_test_dataset(datamodule)
    if len(predictions) != len(test_dataset):
        print(f"Warning: predictions count ({len(predictions)}) != dataset count ({len(test_dataset)})")
    scores = []
    labels = []
    for i in range(len(predictions)):
        pred = predictions[i]
        score = to_scalar(_get_attr(pred, "pred_score"))
        scores.append(score if score is not None else 0.0)
        item = test_dataset[i] if i < len(test_dataset) else None
        if item is not None:
            lbl = _get_attr(item, "gt_label")
            if lbl is None:
                lbl = _get_attr(item, "label")
            labels.append(int(lbl) if lbl is not None else 0)
        else:
            labels.append(0)
    return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int32)


def compute_threshold_youden(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thr = thresholds[best_idx]
    print(f"Youden threshold = {best_thr:.6f} (TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f})")
    return float(best_thr)


def compute_threshold_percentile(scores, labels, percentile=95):
    good_scores = scores[labels == 0]
    if len(good_scores) == 0:
        print("Warning: no good samples found, fallback to median of all scores.")
        return float(np.median(scores))
    thr = float(np.percentile(good_scores, percentile))
    print(f"Percentile ({percentile}th) threshold = {thr:.6f}")
    return thr


def plot_confusion_matrix(ax, labels, scores, threshold, title_suffix=""):
    pred_labels = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, pred_labels)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Good", "Defect"]
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"CM (thr={threshold:.4f}){title_suffix}")
    thresh_color = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh_color else "black", fontsize=14)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    ax.text(0.5, -0.15, f"Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=10)


def plot_evaluation_report(scores, labels, output_path, thr_youden=None, thr_percentile=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("EfficientAD Capsule — Evaluation Charts", fontsize=16, fontweight="bold")
    # ROC
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Image-level)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    # PR
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    ax = axes[0, 1]
    ax.plot(recall, precision, color="purple", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    baseline = np.mean(labels)
    ax.axhline(baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Image-level)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    # Histogram
    ax = axes[0, 2]
    good_scores = scores[labels == 0]
    defect_scores = scores[labels == 1]
    bins = np.linspace(min(scores.min(), 0), max(scores.max(), 1), 50)
    ax.hist(good_scores, bins=bins, alpha=0.6, label="Good", color="green", edgecolor="black")
    ax.hist(defect_scores, bins=bins, alpha=0.6, label="Defect", color="red", edgecolor="black")
    if thr_youden is not None:
        ax.axvline(thr_youden, color="blue", linestyle="--", lw=2, label=f"Youden thr = {thr_youden:.4f}")
    if thr_percentile is not None:
        ax.axvline(thr_percentile, color="magenta", linestyle=":", lw=2, label=f"P95 thr = {thr_percentile:.4f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    # CM Youden
    plot_confusion_matrix(axes[1, 0], labels, scores, thr_youden if thr_youden is not None else 0.5, title_suffix=" Youden")
    # CM P95
    plot_confusion_matrix(axes[1, 1], labels, scores, thr_percentile if thr_percentile is not None else 0.5, title_suffix=" P95")
    # Metrics comparison
    ax = axes[1, 2]
    def get_metrics(labels, scores, thr):
        pred = (scores >= thr).astype(int)
        cm = confusion_matrix(labels, pred).ravel()
        if len(cm) != 4:
            return 0, 0, 0, 0
        tn, fp, fn, tp = cm
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return acc, prec, rec, f1
    youden_metrics = get_metrics(labels, scores, thr_youden) if thr_youden is not None else (0,0,0,0)
    perc_metrics = get_metrics(labels, scores, thr_percentile) if thr_percentile is not None else (0,0,0,0)
    x = np.arange(4)
    width = 0.35
    labels_bar = ["Accuracy", "Precision", "Recall", "F1"]
    bars1 = ax.bar(x - width/2, youden_metrics, width, label=f"Youden ({thr_youden:.3f})", color="#3498db")
    bars2 = ax.bar(x + width/2, perc_metrics, width, label=f"Percentile ({thr_percentile:.3f})", color="#e74c3c")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar)
    ax.set_ylabel("Score")
    ax.set_title("Metrics Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved evaluation report to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("test_img"))
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/capsule.ckpt"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datamodule = MVTecAD(root=args.dataset_root, category="capsule", train_batch_size=1, eval_batch_size=1, num_workers=args.num_workers)
    model = EfficientAd(model_size=EfficientAdModelSize.M)
    engine = Engine()
    scores, labels = collect_predictions(engine, model, datamodule, args.ckpt)
    if args.threshold is not None:
        print(f"Using MANUAL threshold = {args.threshold}")
        thr_youden = args.threshold
        thr_percentile = args.threshold
    else:
        print("Auto-calculating optimal thresholds...")
        thr_youden = compute_threshold_youden(scores, labels)
        thr_percentile = compute_threshold_percentile(scores, labels, percentile=95)
    plot_evaluation_report(scores, labels, output_path=args.output_dir / "evaluation_report.png", thr_youden=thr_youden, thr_percentile=thr_percentile)

if __name__ == "__main__":
    main()
