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

class CustomCallbackClass(app_callback_class):
    def __init__(self):
        super().__init__()
        self.use_frame = True
        
        # Detection thresholds and parameters
        self.fall_history = deque(maxlen=5)
        self.confidence_threshold = 0.4
        
        # Tracking parameters
        self.tracks = defaultdict(lambda: {
            'positions': deque(maxlen=30),
            'head_positions': deque(maxlen=10),
            'fall_scores': deque(maxlen=10)
        })
        self.next_track_id = 0
        self.track_max_distance = 100
        
        # Fall detection parameters
        self.head_drop_threshold = 0.05
        self.fall_detection_active = False
        self.detection_status = "MONITORING"
        self.fall_score = 0
        
    def _get_track_id(self, bbox, width, height):
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
        try:
            # Get head position
            head = points[0]
            head_y = int((head.y() * bbox.height() + bbox.ymin()) * height)
            
            track = self.tracks[track_id]
            track['head_positions'].append(head_y)
            
            if len(track['head_positions']) < 2:
                return False
                
            prev_head_y = track['head_positions'][-2]
            head_drop = (head_y - prev_head_y) / height
            
            is_fall = head_drop > self.head_drop_threshold
            
            self.fall_history.append(is_fall)
            self.fall_detection_active = sum(self.fall_history) >= 2
            
            current_score = min(100, abs(head_drop * 300))
            track['fall_scores'].append(current_score)
            self.fall_score = sum(track['fall_scores']) / len(track['fall_scores'])
            
            if self.fall_detection_active:
                print("\033[91m[FALL DETECTED]\033[0m Fall Score: {:.1f}".format(self.fall_score))
            elif self.fall_score > 50:
                print("\033[93m[WARNING]\033[0m Fall Score: {:.1f}".format(self.fall_score))
            else:
                print("\033[92m[MONITORING]\033[0m Fall Score: {:.1f}".format(self.fall_score))
            
            return is_fall
            
        except Exception as e:
            print(f"Error in fall detection: {e}")
            return False

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
                    track_id = user_data._get_track_id(bbox, width, height)
                    is_falling = user_data.detect_fall(points, bbox, width, height, track_id)

    if user_data.use_frame and frame is not None:
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = CustomCallbackClass()
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
