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
                        
                        # Adjust text position to prevent going off-screen
                        text_y = max(30, y_min - 5)
                        
                        # Add text with background for better visibility
                        def put_text_with_background(img, text, position, scale=0.7):
                            thickness = 2
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
                            
                            # Draw background box
                            cv2.rectangle(img, 
                                        (position[0], position[1] - text_height - 5),
                                        (position[0] + text_width, position[1] + 5),
                                        (0, 0, 0), 
                                        -1)
                            
                            # Draw text
                            cv2.putText(img, text, position, font, scale, (255, 255, 255), thickness)
                        
                        # Display multiple lines of text
                        text_lines = [
                            f"Person: {confidence*100:.1f}%",
                            f"Fall Score: {user_data.fall_score:.1f}",
                            user_data.detection_status
                        ]
                        
                        for i, text in enumerate(text_lines):
                            put_text_with_background(frame, 
                                                   text, 
                                                   (x_min, text_y + i*25))
                        
                        # Set bounding box color based on fall detection status
                        if user_data.fall_detection_active:
                            box_color = (0, 0, 255)  # Red for fall detected
                        elif user_data.fall_score > 50:
                            box_color = (0, 165, 255)  # Orange for warning
                        else:
                            box_color = (0, 255, 0)  # Green for normal
                            
                        # Draw bounding box
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                                    box_color, 2)
                        
                        # Draw keypoints
                        for point in points:
                            x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                            y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                            # Draw keypoints with black outline for visibility
                            cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    if user_data.use_frame and frame is not None:
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create class instance with frame display enabled
    user_data = CustomCallbackClass()
    # Create and run the application
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
