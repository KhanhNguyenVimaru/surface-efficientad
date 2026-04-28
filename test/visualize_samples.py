import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd


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


def normalize_anomaly_map(anomaly_map):
    if anomaly_map is None:
        return None
    if hasattr(anomaly_map, "detach"):
        anomaly_map = anomaly_map.detach().cpu().numpy()
    arr = np.array(anomaly_map)
    arr = np.squeeze(arr)
    if arr.size == 0:
        return None
    arr = arr.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


def resize_array(arr, width, height):
    image = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    resized = image.resize((width, height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def build_overlay(image_np, anomaly_map, alpha=0.45):
    h, w = image_np.shape[:2]
    if anomaly_map.shape[0] != h or anomaly_map.shape[1] != w:
        anomaly_map = resize_array(anomaly_map, w, h)
    colored = cm.get_cmap("jet")(anomaly_map)[..., :3]
    overlay = ((1.0 - alpha) * image_np.astype(np.float32) + alpha * (colored * 255.0)).clip(0, 255).astype(np.uint8)
    return overlay


def squeeze_batch(pred):
    # Unwrap single-element lists/tuples recursively
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


def plot_sample_visualizations(predictions, test_dataset, output_path, max_samples=8):
    num_samples = min(max_samples, len(test_dataset), len(predictions))
    if num_samples == 0:
        print("No samples to visualize.")
        return

    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        item = test_dataset[i]
        image_tensor = _get_attr(item, "image")
        label = _get_attr(item, "gt_label")
        if label is None:
            label = _get_attr(item, "label")

        if hasattr(image_tensor, "numpy"):
            img_np = image_tensor.numpy()
        else:
            img_np = np.array(image_tensor)

        if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))

        if img_np.dtype in (np.float32, np.float64):
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        pred = predictions[i]
        pred_score = to_scalar(
            _get_attr(pred, "pred_score")
        )
        pred_label = to_scalar(
            _get_attr(pred, "pred_label")
        )
        anomaly_map = _get_attr(pred, "anomaly_map")
        anomaly_map_np = normalize_anomaly_map(anomaly_map)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        if anomaly_map_np is not None:
            axes[i, 1].imshow(anomaly_map_np, cmap="jet", vmin=0, vmax=1)
            axes[i, 1].set_title("Anomaly Map")
        else:
            axes[i, 1].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[i, 1].set_title("Anomaly Map")
        axes[i, 1].axis("off")

        if anomaly_map_np is not None:
            overlay = build_overlay(img_np, anomaly_map_np)
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title("Overlay")
        else:
            axes[i, 2].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[i, 2].set_title("Overlay")
        axes[i, 2].axis("off")

        gt_text = "Defect" if label == 1 else "Good" if label == 0 else "Unknown"
        pred_text = "Defect" if pred_label == 1 else "Good" if pred_label == 0 else "Unknown"
        info = (
            f"GT: {gt_text}\nPred: {pred_text}\nScore: {pred_score:.4f}"
            if pred_score is not None
            else f"GT: {gt_text}\nPred: {pred_text}"
        )
        axes[i, 3].text(
            0.5,
            0.5,
            info,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        axes[i, 3].set_xlim(0, 1)
        axes[i, 3].set_ylim(0, 1)
        axes[i, 3].axis("off")
        axes[i, 3].set_title("Prediction")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved sample visualizations to: {output_path}")


def get_test_dataset(datamodule):
    if hasattr(datamodule, "test_data") and datamodule.test_data is not None:
        return datamodule.test_data
    dl = datamodule.test_dataloader()
    if isinstance(dl, list):
        dl = dl[0]
    return dl.dataset


def main():
    parser = argparse.ArgumentParser(
        description="Predict EfficientAD capsule checkpoint and export sample visualizations."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("mvtec_anomaly_detection"),
        help="Path to MVTec AD root folder.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoints/capsule.ckpt"),
        help="Path to checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save output images.",
    )
    parser.add_argument(
        "--max-viz",
        type=int,
        default=8,
        help="Maximum number of test samples to visualize.",
    )
    args = parser.parse_args()

    if not (args.dataset_root / "capsule").is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {args.dataset_root / 'capsule'}")
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = MVTecAD(
        root=args.dataset_root,
        category="capsule",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=args.num_workers,
    )
    model = EfficientAd()
    engine = Engine()

    print("Running predictions for visualization...")
    raw_predictions = engine.predict(model=model, datamodule=datamodule, ckpt_path=args.ckpt)

    if raw_predictions:
        predictions = [squeeze_batch(p) for p in raw_predictions]
        try:
            test_dataset = get_test_dataset(datamodule)
            plot_sample_visualizations(
                predictions,
                test_dataset,
                args.output_dir / "sample_visualizations.png",
                max_samples=args.max_viz,
            )
        except Exception as e:
            print(f"Visualization failed: {e}")
            raise
    else:
        print("No predictions returned for visualization.")


if __name__ == "__main__":
    main()
