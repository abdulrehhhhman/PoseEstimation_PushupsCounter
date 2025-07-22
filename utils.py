# utils.py
import numpy as np

# Keypoint index constants
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

def calculate_angle(p1, p2, p3):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    v1, v2 = p1 - p2, p3 - p2
    dot = np.dot(v1, v2)
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    cos_angle = np.clip(dot / (norm_v1 * norm_v2), -1.0, 1.0)
    return np.arccos(cos_angle) * 180 / np.pi

def calculate_body_alignment_angle(shoulder, hip, ankle):
    return calculate_angle(shoulder, hip, ankle)

def detect_side_view_orientation(keypoints):
    left = keypoints[LEFT_SHOULDER]
    right = keypoints[RIGHT_SHOULDER]
    return abs(left[0] - right[0]) < 50

def check_keypoint_quality(keypoints, required_points):
    for idx in required_points:
        if len(keypoints) <= idx:
            return False
        point = keypoints[idx]
        if np.allclose(point, [0, 0]) or np.any(np.isnan(point)) or np.any(np.isinf(point)):
            return False
    return True
