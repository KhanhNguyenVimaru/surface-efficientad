import subprocess
import sys
from pathlib import Path


def run_script(name, args):
    script_path = Path(__file__).resolve().parent / name
    cmd = [sys.executable, str(script_path)] + args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
    if result.returncode != 0:
        print(f"[ERROR] {name} failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[OK] {name} completed.\n")


def main():
    dataset_root = "test_img"
    ckpt = "checkpoints/capsule.ckpt"
    output_dir = "outputs"

    run_script("test_capsule_ckpt.py", [
        "--dataset-root", dataset_root,
        "--ckpt", ckpt,
    ])

    run_script("test_capsule_ckpt_with_charts.py", [
        "--dataset-root", dataset_root,
        "--ckpt", ckpt,
        "--output-dir", output_dir,
        "--max-viz", "12",
    ])

    run_script("generate_evaluation_charts.py", [
        "--dataset-root", dataset_root,
        "--ckpt", ckpt,
        "--output-dir", output_dir,
    ])

    run_script("visualize_samples.py", [
        "--dataset-root", dataset_root,
        "--ckpt", ckpt,
        "--output-dir", output_dir,
        "--max-viz", "12",
    ])

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print(f"Check results in: {Path(output_dir).resolve()}")
    print("="*60)


if __name__ == "__main__":
    main()
