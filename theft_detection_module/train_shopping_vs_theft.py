import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Sample:
    path: Path
    label: int
    class_name: str


class VideoBinaryDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        positive_names: List[str],
        negative_names: List[str],
        num_frames: int,
        image_size: int,
        use_handcrafted_features: bool,
    ) -> None:
        self.split_dir = split_dir
        self.positive_names = {x.lower() for x in positive_names}
        self.negative_names = {x.lower() for x in negative_names}
        self.num_frames = num_frames
        self.image_size = image_size
        self.use_handcrafted_features = use_handcrafted_features
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.samples = self._discover_samples()
        if not self.samples:
            raise ValueError(f"No videos found under {split_dir}")

    def _discover_samples(self) -> List[Sample]:
        samples: List[Sample] = []
        if not self.split_dir.exists():
            raise ValueError(f"Split directory does not exist: {self.split_dir}")

        for class_dir in sorted([p for p in self.split_dir.iterdir() if p.is_dir()]):
            class_name = class_dir.name
            class_key = class_name.lower()
            if class_key in self.positive_names:
                label = 1
            elif class_key in self.negative_names:
                label = 0
            else:
                continue

            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in VIDEO_EXTENSIONS:
                    samples.append(Sample(path=path, label=label, class_name=class_name))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            return [0] * self.num_frames
        if total_frames < self.num_frames:
            idxs = np.linspace(0, max(total_frames - 1, 0), self.num_frames)
        else:
            idxs = np.linspace(0, total_frames - 1, self.num_frames)
        return [int(round(i)) for i in idxs]

    def _read_video(self, path: Path) -> Tuple[List[Image.Image], np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._sample_frame_indices(total_frames)
        selected = []
        motion_gray_frames = []

        frame_set = set(indices)
        current_idx = 0
        target_ptr = 0
        next_target = indices[target_ptr] if indices else None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if next_target is not None and current_idx == next_target:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                selected.append(pil)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (self.image_size, self.image_size))
                motion_gray_frames.append(gray)
                target_ptr += 1
                if target_ptr >= len(indices):
                    break
                next_target = indices[target_ptr]
            current_idx += 1

        cap.release()

        if not selected:
            # fallback single black frame
            black = Image.fromarray(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
            selected = [black for _ in range(self.num_frames)]
            motion_gray_frames = [np.zeros((self.image_size, self.image_size), dtype=np.uint8) for _ in range(self.num_frames)]

        while len(selected) < self.num_frames:
            selected.append(selected[-1].copy())
            motion_gray_frames.append(motion_gray_frames[-1].copy())

        gray_stack = np.stack(motion_gray_frames[: self.num_frames], axis=0)
        return selected[: self.num_frames], gray_stack

    def _extract_handcrafted_features(self, gray_stack: np.ndarray) -> np.ndarray:
        if gray_stack.shape[0] < 2:
            return np.zeros(4, dtype=np.float32)

        mags = []
        centers = []
        prev = gray_stack[0]
        for i in range(1, gray_stack.shape[0]):
            curr = gray_stack[i]
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mags.append(float(np.mean(mag)))

            motion_mask = mag > np.percentile(mag, 85)
            ys, xs = np.where(motion_mask)
            if len(xs) > 0 and len(ys) > 0:
                centers.append([float(xs.mean()), float(ys.mean())])
            prev = curr

        mean_mag = float(np.mean(mags)) if mags else 0.0
        max_mag = float(np.max(mags)) if mags else 0.0
        burst = float(np.std(mags)) if len(mags) > 1 else 0.0

        if len(centers) > 1:
            centers_arr = np.array(centers, dtype=np.float32)
            spread = float(np.linalg.norm(centers_arr[-1] - centers_arr[0]) / max(self.image_size, 1))
        else:
            spread = 0.0

        return np.array([mean_mag, max_mag, burst, spread], dtype=np.float32)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frames, gray_stack = self._read_video(sample.path)
        frame_tensor = torch.stack([self.transform(frame) for frame in frames], dim=0)
        if self.use_handcrafted_features:
            handcrafted = torch.tensor(self._extract_handcrafted_features(gray_stack), dtype=torch.float32)
        else:
            handcrafted = torch.zeros(4, dtype=torch.float32)
        label = torch.tensor(sample.label, dtype=torch.float32)
        return frame_tensor, handcrafted, label, str(sample.path)


class ResNetGRUClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 256, handcrafted_dim: int = 4, dropout: float = 0.3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + handcrafted_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, frames: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        x = frames.view(b * t, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, t, -1)
        seq_out, _ = self.gru(feats)
        temporal_feat = seq_out.mean(dim=1)
        combined = torch.cat([temporal_feat, handcrafted], dim=1)
        logits = self.classifier(combined).squeeze(1)
        return logits


class Trainer:
    def __init__(self, model, device, lr: float, output_dir: Path):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=2)
        self.output_dir = output_dir

    def run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        losses = []
        all_labels = []
        all_probs = []

        iterator = tqdm(loader, leave=False)
        for frames, handcrafted, labels, _paths in iterator:
            frames = frames.to(self.device)
            handcrafted = handcrafted.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(train):
                logits = self.model(frames, handcrafted)
                loss = self.criterion(logits, labels)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            losses.append(float(loss.item()))
            iterator.set_description(f"{'train' if train else 'eval '} loss={loss.item():.4f}")

        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        metrics = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": accuracy_score(all_labels, preds) if all_labels else 0.0,
            "precision": precision_score(all_labels, preds, zero_division=0) if all_labels else 0.0,
            "recall": recall_score(all_labels, preds, zero_division=0) if all_labels else 0.0,
            "f1": f1_score(all_labels, preds, zero_division=0) if all_labels else 0.0,
        }
        return metrics

    @torch.no_grad()
    def evaluate_full(self, loader: DataLoader) -> Dict[str, object]:
        self.model.eval()
        all_labels, all_probs, all_paths = [], [], []
        for frames, handcrafted, labels, paths in tqdm(loader, leave=False):
            frames = frames.to(self.device)
            handcrafted = handcrafted.to(self.device)
            logits = self.model(frames, handcrafted)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)

        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        cm = confusion_matrix(all_labels, preds, labels=[0, 1])
        report = classification_report(all_labels, preds, target_names=["normal", "theft"], zero_division=0)
        metrics = {
            "accuracy": accuracy_score(all_labels, preds),
            "precision": precision_score(all_labels, preds, zero_division=0),
            "recall": recall_score(all_labels, preds, zero_division=0),
            "f1": f1_score(all_labels, preds, zero_division=0),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "predictions": [
                {"path": p, "label": int(y), "prob_theft": float(prob), "pred": int(pred)}
                for p, y, prob, pred in zip(all_paths, all_labels, all_probs, preds)
            ],
        }
        return metrics


def plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training / Validation Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_f1"], label="train_f1")
    plt.plot(epochs, history["val_f1"], label="val_f1")
    plt.xlabel("epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.title("Training / Validation F1")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()



def plot_confusion(cm: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    classes = ["normal", "theft"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()



def build_loaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = VideoBinaryDataset(
        split_dir=Path(args.data_root) / "train",
        positive_names=args.positive_names,
        negative_names=args.negative_names,
        num_frames=args.num_frames,
        image_size=args.image_size,
        use_handcrafted_features=not args.disable_handcrafted_features,
    )
    val_ds = VideoBinaryDataset(
        split_dir=Path(args.data_root) / "val",
        positive_names=args.positive_names,
        negative_names=args.negative_names,
        num_frames=args.num_frames,
        image_size=args.image_size,
        use_handcrafted_features=not args.disable_handcrafted_features,
    )
    test_ds = VideoBinaryDataset(
        split_dir=Path(args.data_root) / "test",
        positive_names=args.positive_names,
        negative_names=args.negative_names,
        num_frames=args.num_frames,
        image_size=args.image_size,
        use_handcrafted_features=not args.disable_handcrafted_features,
    )

    print("Train distribution:", Counter(s.label for s in train_ds.samples))
    print("Val distribution:", Counter(s.label for s in val_ds.samples))
    print("Test distribution:", Counter(s.label for s in test_ds.samples))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader



def main() -> None:
    parser = argparse.ArgumentParser(description="Train shopping-vs-theft binary classifier.")
    parser.add_argument("--data_root", type=str, required=True, help="Root folder containing train/val/test directories.")
    parser.add_argument("--positive_names", nargs="+", default=["theft", "shoplifting", "stealing"], help="Folder names mapped to label 1.")
    parser.add_argument("--negative_names", nargs="+", default=["normal", "shopping"], help="Folder names mapped to label 0.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--disable_handcrafted_features", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path("runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {output_dir.resolve()}")

    train_loader, val_loader, test_loader = build_loaders(args)

    model = ResNetGRUClassifier()
    trainer = Trainer(model=model, device=device, lr=args.lr, output_dir=output_dir)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_f1 = -math.inf
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = trainer.run_epoch(train_loader, train=True)
        val_metrics = trainer.run_epoch(val_loader, train=False)
        trainer.scheduler.step(val_metrics["f1"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        print("Train:", train_metrics)
        print("Val  :", val_metrics)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best model to {best_path}")

    plot_training_curves(history, output_dir / "training_curves.png")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = trainer.evaluate_full(test_loader)
    cm = np.array(test_metrics["confusion_matrix"], dtype=int)
    plot_confusion(cm, output_dir / "confusion_matrix.png")

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(test_metrics["classification_report"])

    metrics_to_save = {
        "best_val_f1": best_val_f1,
        "history": history,
        "test": {
            k: v for k, v in test_metrics.items() if k not in {"classification_report", "predictions"}
        },
        "args": vars(args),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2)

    with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics["predictions"], f, indent=2)

    print("\nFinal test metrics:")
    print(json.dumps(metrics_to_save["test"], indent=2))
    print("\nClassification report:\n")
    print(test_metrics["classification_report"])


if __name__ == "__main__":
    main()
