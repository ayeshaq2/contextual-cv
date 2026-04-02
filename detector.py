"""
detector.py
-----------
Runs YOLOv8-Pose on a single image frame to detect people and extract
their 17 skeletal keypoints.

Keypoint index reference (COCO format):
    0: nose         1: left eye      2: right eye
    3: left ear     4: right ear     5: left shoulder
    6: right shoulder  7: left elbow  8: right elbow
    9: left wrist  10: right wrist  11: left hip
   12: right hip   13: left knee   14: right knee
   15: left ankle  16: right ankle
"""

from ultralytics import YOLO
import numpy as np
import cv2

# Keypoint indices we care about most
KP = {
    "nose": 0,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7,    "right_elbow": 8,
    "left_wrist": 9,    "right_wrist": 10,
    "left_hip": 11,     "right_hip": 12,
    "left_knee": 13,    "right_knee": 14,
    "left_ankle": 15,   "right_ankle": 16,
}


class PoseDetector:
    def __init__(self, model_size="yolov8n-pose.pt", confidence=0.3):
        """
        Args:
            model_size: YOLOv8 pose model. 'yolov8n-pose.pt' is smallest/fastest.
            confidence: minimum detection confidence threshold.
        """
        print(f"[Detector] Loading {model_size}...")
        self.model = YOLO(model_size)
        self.confidence = confidence
        print("[Detector] Ready.")

    def detect(self, image):
        """
        Run pose estimation on a single frame.

        Args:
            image: BGR numpy array (as loaded by cv2.imread)

        Returns:
            List of person dicts, one per detected person:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "keypoints": numpy array of shape (17, 3)
                             each row is [x, y, confidence]
                             x, y are pixel coordinates
                             (0, 0, 0) means keypoint not detected
            }
        """
        results = self.model(image, verbose=False)[0]
        people = []

        if results.keypoints is None:
            return people

        for i, box in enumerate(results.boxes):
            conf = float(box.conf[0])
            if conf < self.confidence:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Keypoints: shape (17, 3) — x, y, confidence
            kps = results.keypoints.data[i].cpu().numpy()  # (17, 3)

            people.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "keypoints": kps
            })

        return people

    def get_keypoint(self, person, name):
        """
        Get a specific keypoint by name.

        Args:
            person: dict from detect()
            name: keypoint name string (e.g. "nose", "left_wrist")

        Returns:
            (x, y, confidence) tuple, or (0, 0, 0) if not detected
        """
        idx = KP[name]
        return tuple(person["keypoints"][idx])

    def midpoint(self, person, name_a, name_b):
        """
        Compute the midpoint between two keypoints.
        Returns None if either keypoint has zero confidence.
        """
        ax, ay, ac = self.get_keypoint(person, name_a)
        bx, by, bc = self.get_keypoint(person, name_b)
        if ac == 0 or bc == 0:
            return None
        return ((ax + bx) / 2, (ay + by) / 2)

    def draw(self, image, people):
        """
        Draw pose skeletons on an image for visualisation/debugging.

        Args:
            image: BGR numpy array
            people: output from detect()

        Returns:
            Annotated image copy
        """
        img = image.copy()
        skeleton_pairs = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("right_shoulder", "right_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("right_hip", "right_knee"),
            ("left_knee", "left_ankle"),
            ("right_knee", "right_ankle"),
        ]

        for person in people:
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Draw skeleton lines
            for a, b in skeleton_pairs:
                ax, ay, ac = self.get_keypoint(person, a)
                bx, by, bc = self.get_keypoint(person, b)
                if ac > 0 and bc > 0:
                    cv2.line(img, (int(ax), int(ay)), (int(bx), int(by)),
                             (0, 200, 255), 2)

            # Draw keypoints
            for name, idx in KP.items():
                x, y, c = person["keypoints"][idx]
                if c > 0:
                    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

        return img


# --- Quick test ---
# Run this file directly to test pose detection on a single image:
# python detector.py
if __name__ == "__main__":
    import sys

    # Pass an image path as argument, or it will use webcam frame
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
    else:
        # Grab a single webcam frame as fallback
        cap = cv2.VideoCapture(0)
        ret, image = cap.read()
        cap.release()

    if image is None:
        print("Could not load image.")
    else:
        detector = PoseDetector()
        people = detector.detect(image)
        print(f"Detected {len(people)} person(s).")
        annotated = detector.draw(image, people)
        cv2.imshow("Pose Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
