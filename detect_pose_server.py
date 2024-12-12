import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
from collections import deque, defaultdict
from datetime import datetime
from threading import Thread
from queue import Queue
import sys
import signal
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, QTimer
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from pose_estimation_pipeline import GStreamerPoseEstimationApp
from ptz_tracker import PTZTracker
from led_controller import LEDController
from gui_app import create_gui

class GUIThread(Thread):
    def __init__(self, gui_queue):
        super().__init__()
        self.gui_queue = gui_queue
        self.app = None
        self.window = None
        self.running = True
        self.daemon = True

    def run(self):
        self.app = QApplication(sys.argv)
        self.window = create_gui()
        self.window.show()

        def check_queue():
            try:
                while not self.gui_queue.empty():
                    update_type, data = self.gui_queue.get()
                    if update_type == 'status':
                        self.window.update_status(*data)
                    elif update_type == 'metrics':
                        self.window.update_metrics(*data)
                    elif update_type == 'led_status':
                        self.window.update_led_status(*data)
                    elif update_type == 'ptz_status':
                        self.window.update_ptz_status(data)
                    elif update_type == 'log':
                        self.window.add_log_message(data)
            except Exception as e:
                print(f"GUI update error: {e}")

        timer = QTimer()
        timer.timeout.connect(check_queue)
        timer.start(100)

        self.app.exec_()

    def stop(self):
        self.running = False
        if self.app:
            self.app.quit()

class PTZConnectionThread(Thread):
    def __init__(self, ptz_tracker, gui_queue):
        super().__init__()
        self.ptz_tracker = ptz_tracker
        self.gui_queue = gui_queue
        self.daemon = True

    def run(self):
        try:
            if self.ptz_tracker.start_server():
                self.gui_queue.put(('ptz_status', "Connected"))
                self.gui_queue.put(('log', "PTZ camera connected successfully"))
        except Exception as e:
            self.gui_queue.put(('ptz_status', f"Connection failed: {str(e)}"))
            self.gui_queue.put(('log', f"PTZ connection error: {str(e)}"))

class CustomCallbackClass(app_callback_class):
    def __init__(self, gui_queue):
        super().__init__()
        self.use_frame = True
        self.confidence_threshold = 0.4
        self.gui_queue = gui_queue

        # LED controller initialization
        self.led_controller = LEDController("192.168.0.3")
        self.send_gui_update('led_status', ("Connected", "Initialized"))

        # PTZ tracking initialization
        self.ptz_tracker = PTZTracker()
        self.ptz_connection = PTZConnectionThread(self.ptz_tracker, self.gui_queue)
        self.ptz_connection.start()

        # Fall detection initialization
        self.fall_history = deque(maxlen=5)
        self.tracks = defaultdict(lambda: {
            'positions': deque(maxlen=30),
            'head_positions': deque(maxlen=10),
            'fall_scores': deque(maxlen=10),
            'is_fallen': False,
            'last_box_center': None,
            'fall_time': None,
            'last_movement_time': None,
            'verifying_start_time': None,
            # 관절 움직임 추적을 위한 새 변수들
            'arm_positions': deque(maxlen=30),  # 팔 관절 위치 기록
            'leg_positions': deque(maxlen=30),  # 다리 관절 위치 기록
            'movement_history': deque(maxlen=10)  # 시간 윈도우용 움직임 기록
        })
        
        # State management variables
        self.fall_detection_timestamp = None
        self.confirmed_fall = False
        self.fall_track_id = None
        self.movement_threshold = 0.03
        self.fall_confirmation_time = 2.0
        self.recovery_threshold = 0.01
        self.movement_check_interval = 0.5
        
        self.next_track_id = 0
        self.track_max_distance = 100
        self.head_drop_threshold = 0.02
        self.fall_detection_active = False
        self.fall_score = 0
        self.fall_detection_time = None

    def send_gui_update(self, update_type, data):
        try:
            self.gui_queue.put_nowait((update_type, data))
        except:
            pass

    def calculate_limb_movement(self, points, track_id, height):
        """팔과 다리 관절의 움직임을 계산"""
        track = self.tracks[track_id]
        current_time = time.time()
        
        # 관절 위치 추출
        arms = [points[5], points[6], points[7], points[8]]  # 팔꿈치와 손목
        legs = [points[13], points[14], points[15], points[16]]  # 무릎과 발목
        
        current_arms = [(p.x(), p.y()) for p in arms]
        current_legs = [(p.x(), p.y()) for p in legs]
        
        if not track['arm_positions']:
            track['arm_positions'].append(current_arms)
            track['leg_positions'].append(current_legs)
            return 0
            
        # 이전 위치와 현재 위치의 차이 계산
        prev_arms = track['arm_positions'][-1]
        prev_legs = track['leg_positions'][-1]
        
        arm_movement = max(
            np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            for curr, prev in zip(current_arms, prev_arms)
        ) / height
        
        leg_movement = max(
            np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            for curr, prev in zip(current_legs, prev_legs)
        ) / height
        
        # 새로운 위치 저장
        track['arm_positions'].append(current_arms)
        track['leg_positions'].append(current_legs)
        
        # 움직임 이력 저장
        movement = max(arm_movement, leg_movement)
        track['movement_history'].append((current_time, movement))
        
        return movement

    def calculate_movement(self, current_box, track_id, height):
        """기존의 전체 움직임 계산"""
        track = self.tracks[track_id]
        current_center = [(current_box.xmin() + current_box.xmax()) / 2,
                         (current_box.ymin() + current_box.ymax()) / 2]
        
        if track['last_box_center'] is None:
            track['last_box_center'] = current_center
            return 0
        
        movement = np.sqrt((current_center[0] - track['last_box_center'][0]) ** 2 +
                         (current_center[1] - track['last_box_center'][1]) ** 2) / height
        
        track['last_box_center'] = current_center
        track['last_movement_time'] = time.time()
        return movement

    def check_recent_movement(self, track_id, window_size=1.0):
        """시간 윈도우 내의 움직임 확인"""
        track = self.tracks[track_id]
        if not track['movement_history']:
            return False
            
        current_time = time.time()
        threshold = 0.015  # 관절 움직임 임계값
        
        # 윈도우 내의 움직임만 필터링
        recent_movements = [
            movement for timestamp, movement in track['movement_history']
            if current_time - timestamp <= window_size
        ]
        
        if not recent_movements:
            return False
            
        # 최대 움직임이 임계값을 넘으면 움직임이 있다고 판단
        return max(recent_movements) > threshold

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
            current_time = time.time()
            
            # 전체 움직임과 관절 움직임 계산
            movement = self.calculate_movement(bbox, track_id, height)
            limb_movement = self.calculate_limb_movement(points, track_id, height)
            
            # 낙상 확정 상태에서의 회복 검사
            if self.confirmed_fall and track_id == self.fall_track_id:
                # 시간 윈도우 내에서 움직임이 감지되면 회복으로 판단
                if self.check_recent_movement(track_id):
                    self.confirmed_fall = False
                    self.fall_track_id = None
                    self.fall_detection_timestamp = None
                    self.send_gui_update('status', ("MONITORING", None))
                    self.send_gui_update('log', "Movement detected - Fall recovery")
                    self.led_controller.led_off()
                    self.send_gui_update('led_status', ("Connected", "LED OFF"))
                    track['verifying_start_time'] = None
                    return False
                return True

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

            # 낙상 감지 및 확인 과정
            if is_fall and not self.fall_detection_timestamp:
                self.fall_detection_timestamp = current_time
                self.fall_track_id = track_id
                track['verifying_start_time'] = current_time
                track['fall_time'] = current_time
                self.send_gui_update('status', ("VERIFYING", self.fall_confirmation_time))
                self.send_gui_update('log', "Fall detected, verifying...")
            
            elif is_fall and self.fall_detection_timestamp and track_id == self.fall_track_id:
                elapsed_time = current_time - self.fall_detection_timestamp
                remaining_time = self.fall_confirmation_time - elapsed_time
                
                if elapsed_time >= self.fall_confirmation_time and not self.confirmed_fall:
                    self.confirmed_fall = True
                    self.fall_detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.send_gui_update('status', ("FALL_DETECTED", self.fall_detection_time))
                    self.send_gui_update('log', f"Fall confirmed at {self.fall_detection_time}")
                    self.led_controller.led_on()
                    self.send_gui_update('led_status', ("Connected", "LED ON"))
                else:
                    self.send_gui_update('status', ("VERIFYING", remaining_time))

            # 확인 중 움직임 감지 - 시간 윈도우 내 관절 움직임 확인
            if self.fall_detection_timestamp and not self.confirmed_fall and track_id == self.fall_track_id:
                if self.check_recent_movement(track_id):
                    self.fall_detection_timestamp = None
                    self.fall_track_id = None
                    track['verifying_start_time'] = None
                    self.send_gui_update('status', ("MONITORING", None))
                    self.send_gui_update('log', "Movement detected - Not a fall")
                    
            current_score = min(100, abs(head_drop * 500))
            track['fall_scores'].append(current_score)
            self.fall_score = sum(track['fall_scores']) / len(track['fall_scores'])
            
            self.send_gui_update('metrics', (self.fall_score, len(self.tracks)))

            return self.confirmed_fall and track_id == self.fall_track_id

        except Exception as e:
            self.send_gui_update('log', f"Error in fall detection: {str(e)}")
            return False
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    person_detections = []
    for detection in detections:
        if detection.get_label() == "person" and detection.get_confidence() > user_data.confidence_threshold:
            bbox = detection.get_bbox()
            bbox_size = bbox.width() * bbox.height()
            person_detections.append((detection, bbox_size))

    if person_detections:
        largest_detection, _ = max(person_detections, key=lambda x: x[1])
        bbox = largest_detection.get_bbox()

        if user_data.ptz_tracker and user_data.ptz_tracker.is_connected:
            x_movement, y_movement = user_data.ptz_tracker.calculate_movement(bbox, width, height)
            if x_movement != 0 or y_movement != 0:
                user_data.ptz_tracker.send_movement_command(x_movement, y_movement)

        landmarks = largest_detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if len(landmarks) != 0:
            points = landmarks[0].get_points()
            if len(points) >= 17:
                track_id = user_data.get_track_id(bbox, width, height)
                is_falling = user_data.detect_fall(points, bbox, width, height, track_id)

    return Gst.PadProbeReturn.OK

def main():
    gui_queue = Queue()
    
    # GUI 스레드 시작
    gui_thread = GUIThread(gui_queue)
    gui_thread.start()

    # 메인 앱 실행 (GStreamer)
    user_data = CustomCallbackClass(gui_queue)
    app = GStreamerPoseEstimationApp(app_callback, user_data)

    def signal_handler(sig, frame):
        print("\nShutting down...")
        gui_thread.stop()
        gui_thread.join()
        if hasattr(user_data, 'ptz_tracker'):
            user_data.ptz_tracker.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    app.run()

if __name__ == "__main__":
    main()

