"""
loader.py
---------
Loads images and their YOLO-format labels from the dataset.

Dataset structure (from your archive folder):
    archive/
        train/
            images/   <- .jpg frames
            labels/   <- .txt YOLO annotations
        valid/
            images/
            labels/
        test/
            images/
            labels/

Class mapping (from data.yaml):
    0 = Drowning   (anomalous)
    1 = Swimming   (normal)
    2 = Out of water (ignored)
"""

import os
import cv2


# Class mapping
CLASS_NAMES = {0: "Drowning", 1: "Swimming", 2: "Out of water"}
DROWNING_CLASS = 0
SWIMMING_CLASS = 1


def load_split(dataset_root, split="train", max_samples=None):
    """
    Load all images and their labels from a given split.

    Args:
        dataset_root: path to the archive/ folder on your machine
                      e.g. r"C:/Users/YourName/Downloads/archive"
        split: one of "train", "valid", or "test"

    Returns:
        List of dicts:
        {
            "image_path": str,
            "image": numpy array (BGR),
            "labels": list of dicts:
                {
                    "class_id": int,
                    "class_name": str,
                    "x_center": float,  # normalised 0-1
                    "y_center": float,
                    "width": float,
                    "height": float,
                    "bbox_px": [x1, y1, x2, y2]  # pixel coords
                }
        }
    """
    images_dir = os.path.join(dataset_root, split, "images")
    labels_dir = os.path.join(dataset_root, split, "labels")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Could not find images folder at: {images_dir}")

    samples = []
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"[Loader] Found {len(image_files)} images in '{split}' split.")

    if max_samples:
        image_files = image_files[:max_samples]
        print(f"[Loader] Limiting to {max_samples} samples for this run.")

    for fname in image_files:
        image_path = os.path.join(images_dir, fname)
        label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[Loader] Warning: could not read {image_path}, skipping.")
            continue

        h, w = image.shape[:2]

        # Load labels
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])

                    # Convert normalised coords to pixel coords
                    x1 = int((x_center - bw / 2) * w)
                    y1 = int((y_center - bh / 2) * h)
                    x2 = int((x_center + bw / 2) * w)
                    y2 = int((y_center + bh / 2) * h)

                    labels.append({
                        "class_id": class_id,
                        "class_name": CLASS_NAMES.get(class_id, "unknown"),
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": bw,
                        "height": bh,
                        "bbox_px": [x1, y1, x2, y2]
                    })

        samples.append({
            "image_path": image_path,
            "image": image,
            "labels": labels
        })

    return samples


def get_label_for_sample(sample):
    """
    Returns the dominant ground truth label for a sample.
    If any label is Drowning, the whole sample is considered anomalous.

    Returns: "Drowning", "Swimming", "Out of water", or "unlabelled"
    """
    class_ids = [l["class_id"] for l in sample["labels"]]
    if DROWNING_CLASS in class_ids:
        return "Drowning"
    if SWIMMING_CLASS in class_ids:
        return "Swimming"
    if len(class_ids) > 0:
        return CLASS_NAMES.get(class_ids[0], "unknown")
    return "unlabelled"


def draw_ground_truth(image, labels):
    """
    Draw ground truth bounding boxes on an image for debugging.

    Args:
        image: BGR numpy array
        labels: list of label dicts from load_split()

    Returns:
        Annotated image
    """
    colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 165, 0)}
    img = image.copy()
    for lbl in labels:
        x1, y1, x2, y2 = lbl["bbox_px"]
        color = colors.get(lbl["class_id"], (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, lbl["class_name"], (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


# --- Quick test ---
# Update DATASET_ROOT to your archive folder path and run:
# python loader.py
if __name__ == "__main__":
    DATASET_ROOT = r"path\to\your\archive"  # update this to your local dataset path

    samples = load_split(DATASET_ROOT, split="train", max_samples=10)

    # Show first 5 images with ground truth boxes
    for sample in samples[:5]:
        annotated = draw_ground_truth(sample["image"], sample["labels"])
        label = get_label_for_sample(sample)
        print(f"  {os.path.basename(sample['image_path'])} -> {label} "
              f"({len(sample['labels'])} annotations)")
        cv2.imshow(f"Ground Truth: {label}", annotated)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
