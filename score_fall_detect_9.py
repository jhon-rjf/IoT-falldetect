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
        self.use_frame = True  # Enable frame display
        
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
        self.head_drop_threshold = 0.15  # 15% of height for sudden drop detection
        self.fall_detection_active = False
        self.detection_status = "MONITORING"
        self.fall_score = 0
        
    def _get_track_id(self, bbox, width, height):
        """Track objects between frames"""
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
        """Detect falls based on sudden head position drops"""
        try:
            # Get head position
            head = points[0]
            head_y = int((head.y() * bbox.height() + bbox.ymin()) * height)
            
            # Get track info
            track = self.tracks[track_id]
            track['head_positions'].append(head_y)
            
            # Need at least 2 positions to detect sudden drop
            if len(track['head_positions']) < 2:
                return False
                
            # Calculate head position change
            prev_head_y = track['head_positions'][-2]
            head_drop = (head_y - prev_head_y) / height  # Normalize by frame height
            
            # Detect sudden drop in head position
            is_fall = head_drop > self.head_drop_threshold
            
            # Update fall history and score
            self.fall_history.append(is_fall)
            self.fall_detection_active = sum(self.fall_history) >= 3
            
            # Calculate fall score based on head position change
            current_score = min(100, abs(head_drop * 100))
            track['fall_scores'].append(current_score)
            self.fall_score = sum(track['fall_scores']) / len(track['fall_scores'])
            
            # Update detection status
            if self.fall_detection_active:
                self.detection_status = "FALL DETECTED"
            elif self.fall_score > 50:
                self.detection_status = "WARNING"
            else:
                self.detection_status = "MONITORING"
            
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

    # 화면 우측에 상태 패널 추가
    if user_data.use_frame and frame is not None:
        # 패널 배경
        panel_width = 300
        panel_start_x = width - panel_width
        cv2.rectangle(frame, 
                     (panel_start_x, 0), 
                     (width, height), 
                     (0, 0, 0), 
                     -1)
        cv2.rectangle(frame, 
                     (panel_start_x, 0), 
                     (width, height), 
                     (255, 255, 255), 
                     2)
        
        # 패널 제목
        cv2.putText(frame, 
                   "Fall Detection Status", 
                   (panel_start_x + 10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, 
                   (255, 255, 255), 
                   2)

        # 상태 정보 표시 위치 시작점
        status_y = 100

        # 현재 감지 상태에 따른 색상 설정
        if user_data.fall_detection_active:
            status_color = (0, 0, 255)  # Red
            status_bg_color = (0, 0, 100)
        elif user_data.fall_score > 50:
            status_color = (0, 165, 255)  # Orange
            status_bg_color = (0, 82, 127)
        else:
            status_color = (0, 255, 0)  # Green
            status_bg_color = (0, 100, 0)

        # 상태 배경 박스
        cv2.rectangle(frame,
                     (panel_start_x + 10, status_y - 30),
                     (width - 10, status_y + 10),
                     status_bg_color,
                     -1)
        
        # 상태 텍스트
        cv2.putText(frame,
                   user_data.detection_status,
                   (panel_start_x + 20, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.8,
                   status_color,
                   2)

        # Fall Score 표시
        score_y = status_y + 60
        cv2.putText(frame,
                   f"Fall Score: {user_data.fall_score:.1f}",
                   (panel_start_x + 20, score_y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.8,
                   (255, 255, 255),
                   2)

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
                        # Ensure coordinates are within frame bounds
                        x_min = max(0, int(bbox.xmin() * width))
                        y_min = max(0, int(bbox.ymin() * height))
                        x_max = min(int(bbox.xmax() * width), width-1)
                        y_max = min(int(bbox.ymax() * height), height-1)
                        
                        # 상태에 따른 바운딩 박스 색상 설정
                        if user_data.fall_detection_active:
                            box_color = (0, 0, 255)  # Red for fall detected
                        elif user_data.fall_score > 50:
                            box_color = (0, 165, 255)  # Orange for warning
                        else:
                            box_color = (0, 255, 0)  # Green for normal
                            
                        # Draw bounding box
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                                    box_color, 2)
                        
                        # Draw "Person" label above bounding box
                        cv2.putText(frame,
                                  f"Person {confidence*100:.1f}%",
                                  (x_min, y_min - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7,
                                  box_color,
                                  2)
                        
                        # Draw keypoints
                        for point in points:
                            x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                            y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                            cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)  # Black outline
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Green center

    if user_data.use_frame and frame is not None:
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = CustomCallbackClass()
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
