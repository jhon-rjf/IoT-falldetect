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
from ptz_tracker import PTZTracker
from threading import Thread

class CustomCallbackClass(app_callback_class):
    def __init__(self):
        super().__init__()
        self.use_frame = True
        self.confidence_threshold = 0.4
        
        # PTZ tracking initialization
        self.ptz_tracker = PTZTracker()
        Thread(target=self.ptz_tracker.start_server).start()

        # Fall detection initialization
        self.fall_history = deque(maxlen=5)
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
        self.fall_detection_time = None

        # Initialize curses screen
        self.screen = curses.initscr()
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.noecho()
        curses.cbreak()
        self.screen.keypad(1)
        self.screen.nodelay(1)
        curses.curs_set(0)
        self.screen.clear()
        self.screen.refresh()

    def get_track_id(self, bbox, width, height):
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
            track = self.tracks[track_id]
            head = points[0]
            head_y = int((head.y() * bbox.height() + bbox.ymin()) * height)
            track['head_positions'].append(head_y)

            if len(track['head_positions']) < 2:
                return False

            prev_head_y = track['head_positions'][-2]
            head_drop = (head_y - prev_head_y) / height

            if len(track['positions']) >= 2:
                prev_pos = track['positions'][-2]
                curr_pos = track['positions'][-1]
                pos_change = abs(curr_pos[1] - prev_pos[1]) / height
                head_drop = head_drop + (pos_change * 0.5)

            is_fall = head_drop > self.head_drop_threshold
            self.fall_history.append(is_fall)
            was_active = self.fall_detection_active
            self.fall_detection_active = sum(self.fall_history) >= 2

            if self.fall_detection_active and not was_active:
                self.fall_detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
            self.update_display(f"Error in fall detection: {str(e)}")
            return False

    def update_display(self, message=None):
        try:
            self.screen.clear()
            self.screen.addstr(0, 0, "+" + "-" * 58 + "+")
            for i in range(1, 13):
                self.screen.addstr(i, 0, "|")
                self.screen.addstr(i, 59, "|")
            self.screen.addstr(13, 0, "+" + "-" * 58 + "+")

            self.screen.addstr(1, 25, "Fall Detection System")
            self.screen.addstr(3, 2, "Current Status:")

            if self.fall_detection_active:
                self.screen.addstr(4, 2, "[ FALL DETECTED ]", curses.color_pair(1) | curses.A_BOLD)
                if self.fall_detection_time:
                    self.screen.addstr(4, 30, f"at {self.fall_detection_time}",
                                     curses.color_pair(1) | curses.A_BOLD)
            elif self.fall_score > 50:
                self.screen.addstr(4, 2, "[ WARNING ]", curses.color_pair(2) | curses.A_BOLD)
            else:
                self.screen.addstr(4, 2, "[ MONITORING ]", curses.color_pair(3) | curses.A_BOLD)

            self.screen.addstr(5, 2, f"Fall Score: {self.fall_score:.1f}")
            self.screen.addstr(6, 2, f"Active Tracks: {len(self.tracks)}")

            if message:
                self.screen.addstr(11, 2, f"Last Event: {message}")

            self.screen.refresh()
        except Exception as e:
            pass

    def cleanup(self):
        try:
            curses.nocbreak()
            self.screen.keypad(0)
            curses.echo()
            curses.endwin()
        except:
            pass
        if hasattr(self, 'ptz_tracker'):
            self.ptz_tracker.cleanup()

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # 사람 객체들만 필터링
    person_detections = []
    for detection in detections:
        if detection.get_label() == "person" and detection.get_confidence() > user_data.confidence_threshold:
            bbox = detection.get_bbox()
            # 바운딩 박스 크기 계산
            bbox_size = bbox.width() * bbox.height()
            person_detections.append((detection, bbox_size))

    # 바운딩 박스 크기로 정렬하고 가장 큰 것 선택
    if person_detections:
        largest_detection, _ = max(person_detections, key=lambda x: x[1])
        bbox = largest_detection.get_bbox()

        # PTZ tracking
        x_movement, y_movement = user_data.ptz_tracker.calculate_movement(bbox, width, height)
        if x_movement != 0 or y_movement != 0:
            user_data.ptz_tracker.send_movement_command(x_movement, y_movement)

        # Fall detection
        landmarks = largest_detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if len(landmarks) != 0:
            points = landmarks[0].get_points()
            if len(points) >= 17:
                track_id = user_data.get_track_id(bbox, width, height)
                is_falling = user_data.detect_fall(points, bbox, width, height, track_id)

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = None
    try:
        user_data = CustomCallbackClass()
        app = GStreamerPoseEstimationApp(app_callback, user_data)
        app.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if user_data:
            user_data.cleanup()
