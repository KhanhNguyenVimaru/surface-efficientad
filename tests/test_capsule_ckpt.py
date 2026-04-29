import argparse
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.lightning_model import EfficientAdModelSize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EfficientAD capsule checkpoint on MVTec AD capsule test set."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("test_img"),
        help="Path to MVTec AD root folder (contains capsule/).",
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
        help="Dataloader workers for testing.",
    )
    args = parser.parse_args()

    if not (args.dataset_root / "capsule").is_dir():
        if not (args.dataset_root / "good").is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {args.dataset_root}")
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")

    datamodule = MVTecAD(
        root=args.dataset_root,
        category="capsule",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=args.num_workers,
    )
    model = EfficientAd(model_size=EfficientAdModelSize.M)
    engine = Engine()

    metrics = engine.test(model=model, datamodule=datamodule, ckpt_path=args.ckpt)
    result = metrics[0] if metrics else {}

    print("=== Test Result: capsule.ckpt on MVTec AD capsule ===")
    for key in sorted(result):
        print(f"{key}: {result[key]:.6f}")


if __name__ == "__main__":
    main()
