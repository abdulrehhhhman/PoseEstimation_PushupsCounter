import numpy as np
from utils import (
    calculate_angle,
    calculate_body_alignment_angle,
    detect_side_view_orientation,
    check_keypoint_quality,
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP, LEFT_KNEE,
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP, RIGHT_KNEE
)

class EnhancedPushupCounter:
    def __init__(self):
        self.count = 0
        self.direction = 0  # 0 = down, 1 = up
        self.prev_state = 'up'

    def detect_pushup_pose(self, keypoints, orientation):
        if orientation == 'left':
            shoulder = keypoints[LEFT_SHOULDER][:2]
            elbow = keypoints[LEFT_ELBOW][:2]
            wrist = keypoints[LEFT_WRIST][:2]
            hip = keypoints[LEFT_HIP][:2]
            knee = keypoints[LEFT_KNEE][:2]
        else:
            shoulder = keypoints[RIGHT_SHOULDER][:2]
            elbow = keypoints[RIGHT_ELBOW][:2]
            wrist = keypoints[RIGHT_WRIST][:2]
            hip = keypoints[RIGHT_HIP][:2]
            knee = keypoints[RIGHT_KNEE][:2]

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        body_angle = calculate_body_alignment_angle(shoulder, hip, knee)

        down = elbow_angle < 90 and body_angle > 150
        up = elbow_angle > 160 and body_angle > 150

        return 'down' if down else 'up' if up else 'unknown'

    def count_pushup_enhanced(self, keypoints_data_list):
        self.count = 0
        self.direction = 0
        self.prev_state = 'up'

        for keypoints in keypoints_data_list:
            orientation = detect_side_view_orientation(keypoints)
            if not check_keypoint_quality(keypoints, orientation):
                continue

            current_state = self.detect_pushup_pose(keypoints, orientation)

            if self.prev_state == 'up' and current_state == 'down':
                self.direction = 1  # Going down
            elif self.prev_state == 'down' and current_state == 'up' and self.direction == 1:
                self.count += 1
                self.direction = 0  # Reset
            self.prev_state = current_state

        return self.count

def process_video_enhanced(keypoints_list):
    counter = EnhancedPushupCounter()
    return counter.count_pushup_enhanced(keypoints_list)
