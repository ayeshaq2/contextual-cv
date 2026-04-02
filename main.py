"""
main.py
-------
Runs the full swimming vs. drowning detection pipeline.

Usage:
    python main.py --dataset C:/Users/YourName/Downloads/archive --split test

The pipeline:
    1. Load images + ground truth labels from the dataset
    2. Run YOLOv8-Pose on each image to extract skeletal keypoints
    3. Group images into windows of WINDOW_SIZE frames
    4. Extract pose features from each window
    5. Score each window as Swimming or Drowning
    6. Compare against ground truth and report accuracy
"""

import argparse
import os
import cv2

from loader import load_split, get_label_for_sample
from detector import PoseDetector
from features import build_window_data, extract_features_from_window, summarise_features
from scorer import score, print_result

# How many consecutive frames to group into one window for temporal analysis.
# Since this dataset is image frames (not video), we simulate windows by
# grouping N consecutive images together.
WINDOW_SIZE = 5


def run_pipeline(dataset_root, split="test", visualise=False, max_windows=None):
    """
    Run the full pipeline on a dataset split.

    Args:
        dataset_root: path to the archive/ folder
        split:        "train", "valid", or "test"
        visualise:    if True, show annotated images as they're processed
        max_windows:  limit number of windows processed (useful for quick testing)
    """
    print(f"\n{'='*55}")
    print(f"  Swimming vs. Drowning Detection Pipeline")
    print(f"  Dataset: {dataset_root}")
    print(f"  Split:   {split}   |   Window size: {WINDOW_SIZE}")
    print(f"{'='*55}\n")

    # Step 1: Load dataset
    samples = load_split(dataset_root, split=split)
    if not samples:
        print("No samples found. Check your dataset path.")
        return

    # Step 2: Initialise detector
    detector = PoseDetector(model_size="yolov8n-pose.pt", confidence=0.3)

    # Step 3: Group samples into windows and process
    correct = 0
    total = 0
    true_positives = 0   # correctly identified drowning
    false_positives = 0  # wrongly flagged as drowning
    false_negatives = 0  # missed drowning

    num_windows = len(samples) // WINDOW_SIZE
    if max_windows:
        num_windows = min(num_windows, max_windows)

    print(f"Processing {num_windows} windows of {WINDOW_SIZE} frames each...\n")

    for w in range(num_windows):
        window_samples = samples[w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]

        # Determine ground truth for this window
        # If any frame in the window is labelled Drowning, it's a drowning window
        ground_truth = "Swimming"
        for s in window_samples:
            if get_label_for_sample(s) == "Drowning":
                ground_truth = "Drowning"
                break

        # Run pose detection on each frame in the window
        person_sequence = []
        for sample in window_samples:
            people = detector.detect(sample["image"])
            # Take the most confident person per frame
            if people:
                best = max(people, key=lambda p: p["confidence"])
                person_sequence.append(best)
            else:
                person_sequence.append(None)

            # Optional visualisation
            if visualise:
                vis = detector.draw(sample["image"], people)
                cv2.imshow("Pose Detection", vis)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    visualise = False

        # Step 4: Extract features from window
        window_data = build_window_data(person_sequence, detector)
        features = extract_features_from_window(window_data)

        # Step 5: Score the window
        if not features.get("body_angle"):
            print(f"  [DEBUG] features dict: {features}")
        result = score(features)
        predicted = result["label"]
        
        # Track metrics
        total += 1
        if predicted == ground_truth:
            correct += 1
        if ground_truth == "Drowning" and predicted == "Drowning":
            true_positives += 1
        if ground_truth == "Swimming" and predicted == "Drowning":
            false_positives += 1
        if ground_truth == "Drowning" and predicted == "Swimming":
            false_negatives += 1

        # Print window result
        match = "✓" if predicted == ground_truth else "✗"
        print(f"Window {w+1:03d}/{num_windows} | GT: {ground_truth:<10} "
              f"Pred: {predicted:<10} {match}")
        if features.get("valid"):
            summarise_features(features)
        print_result(result)
        print()

    # --- Final evaluation report ---
    if visualise:
        cv2.destroyAllWindows()

    accuracy = correct / total if total > 0 else 0
    precision = (true_positives / (true_positives + false_positives)
                 if (true_positives + false_positives) > 0 else 0)
    recall = (true_positives / (true_positives + false_negatives)
              if (true_positives + false_negatives) > 0 else 0)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    fpr = (false_positives / (false_positives + (total - true_positives - false_negatives))
           if total > 0 else 0)

    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS ({split} split)")
    print(f"{'='*55}")
    print(f"  Windows evaluated: {total}")
    print(f"  Accuracy:          {accuracy:.1%}")
    print(f"  Precision:         {precision:.1%}")
    print(f"  Recall:            {recall:.1%}")
    print(f"  F1 Score:          {f1:.1%}")
    print(f"  False Positive Rate: {fpr:.1%}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Swimming vs. Drowning Detection Pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=r"Path to dataset root folder, e.g. C:\Users\YourName\Downloads\archive"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Which dataset split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Show pose detection visualisation while running"
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Limit number of windows to process (for quick testing)"
    )

    args = parser.parse_args()
    run_pipeline(
        dataset_root=args.dataset,
        split=args.split,
        visualise=args.visualise,
        max_windows=args.max_windows
    )
