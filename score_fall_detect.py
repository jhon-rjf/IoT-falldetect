import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
import curses
from collections import deque, defaultdict
from datetime import datetime
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from pose_estimation_pipeline import GStreamerPoseEstimationApp

class CustomCallbackClass(app_callback_class):
    def __init__(self):
        super().__init__()
        self.last_video_end_time = 0
        self.video_restart_delay = 5
        
        # Initialize state variables
        self.use_frame = True
        self.fall_history = deque(maxlen=5)
        self.confidence_threshold = 0.4
        self.tracks = defaultdict(lambda: {
            'positions': deque(maxlen=30),
            'head_positions': deque(maxlen=10),
            'fall_scores': deque(maxlen=10),
            'is_fallen': False
        })
        self.next_track_id = 0
        self.track_max_distance = 100
        self.head_drop_threshold = 0.02
        self.fall_detection_active = False
        self.detection_status = "MONITORING"
        self.fall_score = 0
        self.fall_frame = None
        self.fall_detection_time = None
        
        # Video recording variables
        self.frame_buffer = deque(maxlen=120)  # 60fps * 2 seconds = 120 frames
        self.recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.frames_to_record = 240  # 60fps * 4 seconds = 240 frames (2 seconds before and after)
        self.recorded_frames = 0
        
        # Initialize curses screen
        self.screen = curses.initscr()
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
        self.screen.clear()
        
    def cleanup(self):
        """Clean up curses and close video writer if exists"""
        if self.video_writer:
            self.video_writer.release()
        curses.endwin()
        
    def reset_state(self):
        """Reset all states when restarting video"""
        self.fall_history.clear()
        self.tracks.clear()
        self.next_track_id = 0
        self.fall_detection_active = False
        self.detection_status = "MONITORING"
        self.fall_score = 0
        self.fall_frame = None
        self.fall_detection_time = None
        self.frame_buffer.clear()
        self.recording = False
        self.update_display("[System] Resetting all detection states...")
        
    def add_timestamp_to_frame(self, frame):
        """Add timestamp to frame"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
        
    def start_recording(self, frame):
        """Start recording video with timestamp"""
        if not self.recording:
            self.recording = True
            self.recorded_frames = 0
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fall_detection_{current_time}.avi"
            
            height, width = frame.shape[:2]
            self.video_writer = cv2.VideoWriter(
                filename,
                cv2.VideoWriter_fourcc(*'XVID'),
                60.0,  # FPS
                (width, height)
            )
            
            # Write buffered frames first
            for buffered_frame in self.frame_buffer:
                self.video_writer.write(self.add_timestamp_to_frame(buffered_frame))
                self.recorded_frames += 1
    
    def update_display(self, message=None):
        """Update the terminal display"""
        self.screen.clear()
        
        # Draw border
        self.screen.addstr(0, 0, "+" + "-" * 58 + "+")
        for i in range(1, 13):
            self.screen.addstr(i, 0, "|")
            self.screen.addstr(i, 59, "|")
        self.screen.addstr(13, 0, "+" + "-" * 58 + "+")
        
        # Title
        self.screen.addstr(1, 25, "Fall Detection System")
        
        # Status information
        self.screen.addstr(3, 2, "Current Status:")
        
        # Different colors for different states
        if self.fall_detection_active:
            self.screen.addstr(4, 2, "[ FALL DETECTED ]", curses.color_pair(1) | curses.A_BOLD)
            if self.fall_frame:
                self.screen.addstr(4, 30, f"at frame {self.fall_frame}", curses.color_pair(1) | curses.A_BOLD)
        elif self.fall_score > 50:
            self.screen.addstr(4, 2, "[ WARNING ]", curses.color_pair(2) | curses.A_BOLD)
        else:
            self.screen.addstr(4, 2, "[ MONITORING ]", curses.color_pair(3) | curses.A_BOLD)
            
        self.screen.addstr(5, 2, f"Fall Score: {self.fall_score:.1f}")
        self.screen.addstr(6, 2, f"Active Tracks: {len(self.tracks)}")
        
        # Display detection time if available
        if self.fall_detection_time:
            self.screen.addstr(7, 2, f"Last Fall Detected: {self.fall_detection_time}")
            
        # Recording status
        if self.recording:
            self.screen.addstr(8, 2, "Recording: Active", curses.color_pair(1))
        
        # Latest message
        if message:
            self.screen.addstr(11, 2, f"Last Event: {message}")
        
        self.screen.refresh()

    def detect_fall(self, points, bbox, width, height, track_id):
        try:
            track = self.tracks[track_id]
            
            head = points[0]
            head_y = int((head.y() * bbox.height() + bbox.ymin()) * height)
            
            track['head_positions'].append(head_y)
            
            if len(track['head_positions']) < 2:
                return False
                
            prev_head_y = track['head_positions'][-2]
            head_drop = (head_y - prev_head_y) / height
            
            bbox_height = bbox.ymax() - bbox.ymin()
            if len(track['positions']) >= 2:
                prev_pos = track['positions'][-2]
                curr_pos = track['positions'][-1]
                pos_change = abs(curr_pos[1] - prev_pos[1]) / height
                head_drop = head_drop + (pos_change * 0.5)
            
            is_fall = head_drop > self.head_drop_threshold
            
            self.fall_history.append(is_fall)
            was_active = self.fall_detection_active
            self.fall_detection_active = sum(self.fall_history) >= 2
            
            # Record fall detection time and start recording when fall is first detected
            if self.fall_detection_active and not was_active:
                self.fall_frame = self.frame_count
                self.fall_detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if hasattr(self, 'current_frame') and self.current_frame is not None:
                    self.start_recording(self.current_frame)
            
            current_score = min(100, abs(head_drop * 500))
            track['fall_scores'].append(current_score)
            self.fall_score = sum(track['fall_scores']) / len(track['fall_scores'])
            
            if self.fall_detection_active:
                track['is_fallen'] = True
                self.update_display(f"Fall detected at {self.fall_detection_time}!")
            elif self.fall_score > 50:
                self.update_display("Warning: High fall risk")
            else:
                self.update_display("Normal monitoring")
            
            return is_fall
            
        except Exception as e:
            self.update_display(f"Error in fall detection: {e}")
            return False

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)

    if user_data.frame_count == 1:
        user_data.update_display("Waiting 5 seconds before starting next video...")
        time.sleep(5)
        user_data.reset_state()

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add timestamp to frame
        frame_with_timestamp = user_data.add_timestamp_to_frame(frame.copy())
        
        # Store frame in buffer
        user_data.frame_buffer.append(frame.copy())
        user_data.current_frame = frame.copy()
        
        # If recording, write frame with timestamp
        if user_data.recording:
            user_data.video_writer.write(frame_with_timestamp)
            user_data.recorded_frames += 1
            
            # Stop recording after desired duration
            if user_data.recorded_frames >= user_data.frames_to_record:
                user_data.recording = False
                user_data.video_writer.release()
                user_data.video_writer = None

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

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = None
    try:
        user_data = CustomCallbackClass()
        app = GStreamerPoseEstimationApp(app_callback, user_data)
        app.run()
    finally:
        if user_data:
            user_data.cleanup()
