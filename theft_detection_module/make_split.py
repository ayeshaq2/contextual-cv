import argparse
import random
import shutil
from pathlib import Path
from typing import List

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}


def copy_files(files: List[Path], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dest_dir / src.name)



def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test split from class folders.")
    parser.add_argument("--input_root", required=True, help="Folder with one subfolder per class.")
    parser.add_argument("--output_root", required=True, help="Output folder containing train/val/test.")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        videos = [p for p in class_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
        random.shuffle(videos)

        n = len(videos)
        n_test = int(round(n * args.test_ratio))
        n_val = int(round(n * args.val_ratio))
        n_train = max(0, n - n_val - n_test)

        train_files = videos[:n_train]
        val_files = videos[n_train:n_train + n_val]
        test_files = videos[n_train + n_val:]

        copy_files(train_files, output_root / "train" / class_dir.name)
        copy_files(val_files, output_root / "val" / class_dir.name)
        copy_files(test_files, output_root / "test" / class_dir.name)

        print(f"{class_dir.name}: train={len(train_files)} val={len(val_files)} test={len(test_files)}")


if __name__ == "__main__":
    main()
