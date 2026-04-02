"""
scorer.py
---------
Rule-based anomaly scorer. Takes the features extracted by features.py
and decides whether the behaviour in a window of frames is normal
(swimming) or anomalous (drowning).

Each rule contributes a score. If the total score exceeds a threshold,
the window is flagged as drowning.
"""


# --- Tunable thresholds ---
# These can be adjusted after testing on your dataset.
THRESHOLDS = {
    "body_angle":           40.0,   # degrees — above this = vertical = drowning signal
    "limb_variance":        15.0,   # pixels  — above this = erratic limbs
    "vertical_disp_rate":   10.0,   # pixels  — above this = oscillating vertically
    "head_elevation_ratio": -5.0,   # pixels  — below this = head tilted back
    "horizontal_progress":  10.0,   # pixels  — below this = not moving forward
    "horizontal_veto":      250.0,  # pixels  — above this = definitely swimming, never flag
    "anomaly_score":         1,     # number of rules triggered to flag as drowning
}


def score(features):
    """
    Apply rule-based scoring to a feature dict.

    Args:
        features: dict from features.extract_features_from_window()

    Returns:
        Dict:
        {
            "anomaly_score":    int,    # number of rules triggered (0-5)
            "is_drowning":      bool,   # True if score >= threshold
            "rules_triggered":  list,   # which rules fired
            "label":            str     # "Drowning" or "Swimming"
        }
    """
    if not features.get("valid"):
        return {
            "anomaly_score": 2,
            "is_drowning": True,
            "rules_triggered": ["No pose detected — person may be submerged or in distress"],
            "label": "Drowning"
        }

    # Veto: if the person is moving significantly horizontally,
    # they are actively swimming — never flag as drowning.
    if features["horizontal_progress"] > THRESHOLDS["horizontal_veto"]:
        return {
            "anomaly_score": 0,
            "is_drowning": False,
            "rules_triggered": ["Vetoed: high horizontal progress indicates active swimming"],
            "label": "Swimming"
        }

    rules_triggered = []

    # Rule 0: No pose detected at all — person may be submerged
    body_angle = features.get("body_angle")
    limb_var = features.get("limb_variance")
    if body_angle is None and limb_var is None:
        return {
            "anomaly_score": 2,
            "is_drowning": True,
            "rules_triggered": ["No pose detected — person may be submerged or in distress"],
            "label": "Drowning"
        }

    # Early trigger: body is extremely vertical (>75°) on its own — strong drowning signal
    if features.get("body_angle", 0) > 75.0:
        rules_triggered.append(
            f"Body angle {features.get('body_angle', 0):.1f}° > 75.0° (extreme vertical)"
        )

    # Rule 1: Body is too vertical
    if features["body_angle"] > THRESHOLDS["body_angle"]:
        rules_triggered.append(
            f"Body angle {features['body_angle']:.1f}° > "
            f"{THRESHOLDS['body_angle']}°"
        )

    # Rule 2: Limb movement is erratic
    if features["limb_variance"] > THRESHOLDS["limb_variance"]:
        rules_triggered.append(
            f"Limb variance {features['limb_variance']:.1f}px > "
            f"{THRESHOLDS['limb_variance']}px"
        )

    # Rule 3: Significant vertical oscillation with no forward progress
    if (features["vertical_disp_rate"] > THRESHOLDS["vertical_disp_rate"] and
            features["horizontal_progress"] < THRESHOLDS["horizontal_progress"]):
        rules_triggered.append(
            f"Vertical oscillation {features['vertical_disp_rate']:.1f}px "
            f"with low horizontal progress {features['horizontal_progress']:.1f}px"
        )

    # Rule 4: Head tilted back (low elevation ratio)
    if features["head_elevation_ratio"] < THRESHOLDS["head_elevation_ratio"]:
        rules_triggered.append(
            f"Head elevation ratio {features['head_elevation_ratio']:.1f}px "
            f"< {THRESHOLDS['head_elevation_ratio']}px"
        )

    # Rule 5: No horizontal progress at all
    if features["horizontal_progress"] < THRESHOLDS["horizontal_progress"]:
        rules_triggered.append(
            f"Horizontal progress {features['horizontal_progress']:.1f}px "
            f"< {THRESHOLDS['horizontal_progress']}px"
        )

    anomaly_score = len(rules_triggered)
    is_drowning = anomaly_score >= THRESHOLDS["anomaly_score"]

    return {
        "anomaly_score": anomaly_score,
        "is_drowning": is_drowning,
        "rules_triggered": rules_triggered,
        "label": "Drowning" if is_drowning else "Swimming"
    }


def print_result(result):
    """Print a human-readable scoring result."""
    status = "🚨 DROWNING DETECTED" if result["is_drowning"] else "✓  Normal swimming"
    print(f"  {status}  (score: {result['anomaly_score']}/5)")
    for rule in result["rules_triggered"]:
        print(f"    ↳ {rule}")