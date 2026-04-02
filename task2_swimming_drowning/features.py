"""
features.py
-----------
Computes pose-based features from a sequence of detected persons across
frames. These features are what distinguish drowning from swimming:

    - Body orientation angle  (vertical = drowning, horizontal = swimming)
    - Limb movement variance  (erratic = drowning, rhythmic = swimming)
    - Vertical displacement   (oscillating vertically = drowning)
    - Head elevation ratio    (head tilted back = drowning)
    - Horizontal progress     (no forward movement = drowning)
"""

import numpy as np
import math


def _angle_from_horizontal(p1, p2):
    """
    Compute the angle (in degrees) of the line from p1 to p2
    relative to the horizontal axis.
    Returns value in [0, 90] — 0 is horizontal, 90 is vertical.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return 0.0
    angle = abs(math.degrees(math.atan2(dy, dx)))
    # Normalise to 0-90 range
    if angle > 90:
        angle = 180 - angle
    return angle


def extract_features_from_window(window_data):
    """
    Extract drowning-relevant features from a window of frames.

    Args:
        window_data: list of per-frame dicts (output of build_window_data()).
                     Each dict contains keypoint positions for one frame.
                     See build_window_data() below.

    Returns:
        Dict of features:
        {
            "body_angle":           float,  # degrees, 90=vertical(drowning)
            "limb_variance":        float,  # std dev of wrist/ankle motion
            "vertical_disp_rate":   float,  # avg vertical oscillation
            "head_elevation_ratio": float,  # nose y relative to shoulders
            "horizontal_progress":  float,  # net x movement across window
            "valid":                bool    # False if not enough keypoints
        }
    """
    angles = []
    wrist_positions = []   # list of (lx, ly, rx, ry) per frame
    ankle_positions = []
    centroid_x = []
    centroid_y = []
    head_ratios = []

    for frame in window_data:
        # --- Body orientation angle ---
        shoulder_mid = frame.get("shoulder_mid")
        hip_mid = frame.get("hip_mid")
        if shoulder_mid and hip_mid:
            angle = _angle_from_horizontal(shoulder_mid, hip_mid)
            angles.append(angle)

            # Centroid as midpoint between shoulder_mid and hip_mid
            cx = (shoulder_mid[0] + hip_mid[0]) / 2
            cy = (shoulder_mid[1] + hip_mid[1]) / 2
            centroid_x.append(cx)
            centroid_y.append(cy)

        # --- Wrist and ankle positions for limb variance ---
        lw = frame.get("left_wrist")
        rw = frame.get("right_wrist")
        la = frame.get("left_ankle")
        ra = frame.get("right_ankle")

        if lw and rw:
            wrist_positions.append((lw[0], lw[1], rw[0], rw[1]))
        if la and ra:
            ankle_positions.append((la[0], la[1], ra[0], ra[1]))

        # --- Head elevation ratio ---
        nose = frame.get("nose")
        if nose and shoulder_mid:
            # Positive = nose above shoulder midpoint (normal)
            # Negative = nose below shoulder midpoint (unusual / head back)
            ratio = (shoulder_mid[1] - nose[1])
            head_ratios.append(ratio)

    # --- Compute features ---
    features = {"valid": False}

    if len(angles) < 2:
        return features  # not enough data in this window

    features["valid"] = True

    # Average body orientation angle across window
    features["body_angle"] = float(np.mean(angles))

    # Limb movement variance — std dev of displacement between frames
    limb_disps = []
    for positions in [wrist_positions, ankle_positions]:
        if len(positions) >= 2:
            arr = np.array(positions)
            diffs = np.diff(arr, axis=0)
            displacements = np.linalg.norm(diffs, axis=1)
            limb_disps.extend(displacements.tolist())

    features["limb_variance"] = float(np.std(limb_disps)) if limb_disps else 0.0

    # Vertical displacement rate — std dev of centroid y across window
    # High std dev = lots of vertical oscillation = drowning signal
    features["vertical_disp_rate"] = (
        float(np.std(centroid_y)) if len(centroid_y) >= 2 else 0.0
    )

    # Head elevation ratio — average
    features["head_elevation_ratio"] = (
        float(np.mean(head_ratios)) if head_ratios else 0.0
    )

    # Horizontal progress — net x displacement of centroid across window
    # Small value = not moving forward = drowning signal
    features["horizontal_progress"] = (
        float(abs(centroid_x[-1] - centroid_x[0]))
        if len(centroid_x) >= 2 else 0.0
    )

    return features


def build_window_data(person_sequence, detector):
    """
    Convert a list of 'person' dicts (from detector.detect()) into
    the structured per-frame format used by extract_features_from_window().

    Args:
        person_sequence: list of person dicts, one per frame in the window.
                         Pass None for frames where no person was detected.
        detector: PoseDetector instance (used to call get_keypoint/midpoint)

    Returns:
        List of per-frame dicts ready for extract_features_from_window()
    """
    window_data = []

    for person in person_sequence:
        if person is None:
            window_data.append({})
            continue

        def kp(name):
            x, y, c = detector.get_keypoint(person, name)
            return (x, y) if c > 0 else None

        frame_data = {
            "nose":          kp("nose"),
            "left_shoulder": kp("left_shoulder"),
            "right_shoulder":kp("right_shoulder"),
            "left_elbow":    kp("left_elbow"),
            "right_elbow":   kp("right_elbow"),
            "left_wrist":    kp("left_wrist"),
            "right_wrist":   kp("right_wrist"),
            "left_hip":      kp("left_hip"),
            "right_hip":     kp("right_hip"),
            "left_knee":     kp("left_knee"),
            "right_knee":    kp("right_knee"),
            "left_ankle":    kp("left_ankle"),
            "right_ankle":   kp("right_ankle"),
            "shoulder_mid":  detector.midpoint(person, "left_shoulder", "right_shoulder"),
            "hip_mid":       detector.midpoint(person, "left_hip", "right_hip"),
        }
        window_data.append(frame_data)

    return window_data


def summarise_features(features):
    """
    Print a human-readable summary of extracted features.
    """
    if not features.get("valid"):
        print("  [Features] Not enough data to compute features.")
        return
    print(f"  Body angle:           {features['body_angle']:.1f}° "
          f"({'HIGH - vertical' if features['body_angle'] > 50 else 'normal'})")
    print(f"  Limb variance:        {features['limb_variance']:.2f} px "
          f"({'HIGH - erratic' if features['limb_variance'] > 20 else 'normal'})")
    print(f"  Vertical disp rate:   {features['vertical_disp_rate']:.2f} px "
          f"({'HIGH - oscillating' if features['vertical_disp_rate'] > 15 else 'normal'})")
    print(f"  Head elevation ratio: {features['head_elevation_ratio']:.2f} px")
    print(f"  Horizontal progress:  {features['horizontal_progress']:.2f} px "
          f"({'LOW - not moving' if features['horizontal_progress'] < 10 else 'normal'})")
