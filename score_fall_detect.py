import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from collections import deque, defaultdict
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from pose_estimation_pipeline import GStreamerPoseEstimationApp

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # Fall detection parameters
        self.height_ratio_threshold = 0.5
        self.angle_threshold = 60
        self.fall_history = deque(maxlen=5)
        
        # Detection confidence threshold
        self.confidence_threshold = 0.4
        
        # Tracking parameters
        self.tracks = defaultdict(lambda: {'positions': deque(maxlen=30), 
                                         'fall_scores': deque(maxlen=10)})
        self.next_track_id = 0
        self.track_max_distance = 100
        
        # Metrics
        self.current_height_ratio = 0
        self.current_body_angle = 0
        self.fall_score = 0

    def calculate_height_ratio(self, points, bbox, width, height):
        """Calculate height ratio with improved stability"""
        try:
            head = points[0]
            left_ankle = points[15]
            right_ankle = points[16]
            
            head_x = int((head.x() * bbox.width() + bbox.xmin()) * width)
            head_y = int((head.y() * bbox.height() + bbox.ymin()) * height)
            left_ankle_x = int((left_ankle.x() * bbox.width() + bbox.xmin()) * width)
            left_ankle_y = int((left_ankle.y() * bbox.height() + bbox.ymin()) * height)
            right_ankle_x = int((right_ankle.x() * bbox.width() + bbox.xmin()) * width)
            right_ankle_y = int((right_ankle.y() * bbox.height() + bbox.ymin()) * height)
            
            ankle_mid_x = (left_ankle_x + right_ankle_x) / 2
            ankle_mid_y = (left_ankle_y + right_ankle_y) / 2
            
            vertical_dist = abs(head_y - ankle_mid_y)
            horizontal_dist = abs(head_x - ankle_mid_x)
            
            if horizontal_dist < 1:
                return self.current_height_ratio
                
            ratio = vertical_dist / horizontal_dist
            
            if abs(ratio - self.current_height_ratio) > 2:
                ratio = self.current_height_ratio * 0.7 + ratio * 0.3
                
            return ratio
        except Exception:
            return self.current_height_ratio

    def calculate_body_angle(self, points, bbox, width, height):
        """Calculate angle of upper body with error handling"""
        try:
            neck = points[1]
            left_hip = points[11]
            right_hip = points[12]
            
            neck_x = int((neck.x() * bbox.width() + bbox.xmin()) * width)
            neck_y = int((neck.y() * bbox.height() + bbox.ymin()) * height)
            left_hip_x = int((left_hip.x() * bbox.width() + bbox.xmin()) * width)
            left_hip_y = int((left_hip.y() * bbox.height() + bbox.ymin()) * height)
            right_hip_x = int((right_hip.x() * bbox.width() + bbox.xmin()) * width)
            right_hip_y = int((right_hip.y() * bbox.height() + bbox.ymin()) * height)
            
            hip_mid_x = (left_hip_x + right_hip_x) / 2
            hip_mid_y = (left_hip_y + right_hip_y) / 2
            
            dx = hip_mid_x - neck_x
            dy = hip_mid_y - neck_y
            angle = abs(np.degrees(np.arctan2(dx, dy)))
            
            if abs(angle - self.current_body_angle) > 30:
                angle = self.current_body_angle * 0.7 + angle * 0.3
            
            return angle
        except Exception:
            return self.current_body_angle

    def get_track_id(self, bbox, width, height):
        """Track objects across frames"""
        center_x = int((bbox.xmin() + bbox.xmax()) * width / 2)
        center_y = int((bbox.ymin() + bbox.ymax()) * height / 2)
        current_pos = np.array([center_x, center_y])
        
        min_dist = float('inf')
        best_track_id = None
        
        for track_id, track_info in self.tracks.items():
            if track_info['positions']:
                last_pos = np.array(track_info['positions'][-1])
                dist = np.linalg.norm(current_pos - last_pos)
                if dist < min_dist and dist < self.track_max_distance:
                    min_dist = dist
                    best_track_id = track_id
        
        if best_track_id is None:
            best_track_id = self.next_track_id
            self.next_track_id += 1
            
        self.tracks[best_track_id]['positions'].append(current_pos)
        return best_track_id

    def detect_fall(self, points, bbox, width, height, track_id):
        """Enhanced fall detection with tracking"""
        self.current_height_ratio = self.calculate_height_ratio(points, bbox, width, height)
        self.current_body_angle = self.calculate_body_angle(points, bbox, width, height)
        
        is_fall = (self.current_height_ratio < self.height_ratio_threshold or 
                  self.current_body_angle > self.angle_threshold)
        
        self.fall_history.append(is_fall)
        
        height_score = max(0, min(100, (1 - self.current_height_ratio) * 100))
        angle_score = max(0, min(100, (self.current_body_angle / 90) * 100))
        current_score = max(height_score, angle_score)
        
        track_scores = self.tracks[track_id]['fall_scores']
        track_scores.append(current_score)
        self.fall_score = sum(track_scores) / len(track_scores)
        
        return sum(self.fall_history) >= 3

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        if label == "person" and confidence > user_data.confidence_threshold:
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                
                if len(points) >= 17:
                    track_id = user_data.get_track_id(bbox, width, height)
                    is_falling = user_data.detect_fall(points, bbox, width, height, track_id)
                    
                    if user_data.use_frame:
                        x_min = int(bbox.xmin() * width)
                        y_min = int(bbox.ymin() * height)
                        x_max = int(bbox.xmax() * width)
                        y_max = int(bbox.ymax() * height)
                        
                        # Display FDS score
                        cv2.putText(frame, f"FDS: {user_data.fall_score:.1f}",
                                  (x_min, y_min - 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                        cv2.putText(frame, f"FDS: {user_data.fall_score:.1f}",
                                  (x_min, y_min - 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Display fall detection warning
                        if is_falling:
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                                        (0, 165, 255), 2)
                            cv2.putText(frame, "FALLDOWN DETECT", 
                                      (x_min, y_min - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                
                # Draw keypoints
                if user_data.use_frame:
                    for point in points:
                        x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                        y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                        cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    if user_data.use_frame:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
