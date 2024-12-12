import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
from video_recorder import FallVideoRecorder
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
from fall_report_client import FallReportClient
from arduino_communication import ArduinoCommunication
from config import Config
from threading import Thread
import socket

class ReportThread(Thread):
    def __init__(self, report_client, arduino_comm, gui_queue):
        super().__init__()
        self.report_client = report_client
        self.arduino_comm = arduino_comm
        self.gui_queue = gui_queue
        self.daemon = True

    def run(self):
        print("[DEBUG] Report thread starting")
        try:
            print("[DEBUG] Sending fall report")
            report_result = self.report_client.send_fall_report()
            print(f"[DEBUG] Report sent, result: {report_result}")
            
            if report_result and report_result.get('password'):
                print("[DEBUG] Sending password to Arduino")
                self.arduino_comm.send_password(report_result['password'])
                print("[DEBUG] Arduino communication complete")
                self.gui_queue.put(('log', "Report sent and password transferred to Arduino"))
            else:
                print("[DEBUG] Report failed or no password received")
        except Exception as e:
            print(f"[DEBUG] Error in report thread: {str(e)}")
            self.gui_queue.put(('log', f"Error in fall reporting: {str(e)}"))
        finally:
            print("[DEBUG] Report thread finished")
            
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

        self.report_client = FallReportClient()
        self.arduino_comm = ArduinoCommunication()

        self.led_controller = LEDController("192.168.0.7")
        self.send_gui_update('led_status', ("Connected", "Initialized"))

        self.ptz_tracker = PTZTracker()
        self.ptz_connection = PTZConnectionThread(self.ptz_tracker, self.gui_queue)
        self.ptz_connection.start()
       

        self.tracks = defaultdict(lambda: {
            'positions': deque(maxlen=30),
            'head_positions': deque(maxlen=10),
            'fall_scores': deque(maxlen=10),
            'is_fallen': False,
            'last_box_center': None,
            'fall_time': None,
            'last_movement_time': None,
            'verifying_start_time': None,
            'wrist_positions': deque(maxlen=10),
        })
        
        self.video_recorder = FallVideoRecorder(gui_queue)
        
        self.fall_detection_timestamp = None
        self.confirmed_fall = False
        self.fall_track_id = None
        self.movement_threshold = 0.03
        self.fall_confirmation_time = 5.0
        self.wrist_movement_threshold = 0.015
        self.head_drop_threshold = 0.02
        self.next_track_id = 0
        self.track_max_distance = 100
        self.movement_check_delay = 2.0
        self.fall_score = 0
        self.fall_detection_time = None

    def __del__(self):
        if hasattr(self, 'report_client'):
            self.report_client.cleanup()

    def send_gui_update(self, update_type, data):
        try:
            self.gui_queue.put_nowait((update_type, data))
        except:
            pass

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

    def calculate_movement(self, current_box, track_id, height):
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

    def calculate_wrist_movement(self, points, track_id, height):
        track = self.tracks[track_id]
        
        wrists = [points[7], points[8]]
        current_wrists = [(p.x(), p.y()) for p in wrists]
        
        if not track['wrist_positions']:
            track['wrist_positions'].append(current_wrists)
            return 0
            
        prev_wrists = track['wrist_positions'][-1]
        
        wrist_movement = max(
            np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            for curr, prev in zip(current_wrists, prev_wrists)
        ) / height
        
        track['wrist_positions'].append(current_wrists)
        
        return wrist_movement

    def detect_fall(self, points, bbox, width, height, track_id):
        try:
            track = self.tracks[track_id]
            current_time = time.time()

            movement = self.calculate_movement(bbox, track_id, height)
            
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

            current_score = min(100, abs(head_drop * 500))
            track['fall_scores'].append(current_score)
            self.fall_score = sum(track['fall_scores']) / len(track['fall_scores'])

            is_fall = head_drop > self.head_drop_threshold

            if (self.fall_detection_timestamp or self.confirmed_fall):
                time_since_detection = current_time - (self.fall_detection_timestamp or 0)
                if time_since_detection > self.movement_check_delay and self.fall_score >= 5:
                    self.fall_detection_timestamp = None
                    self.fall_track_id = None
                    self.confirmed_fall = False
                    track['verifying_start_time'] = None
                    self.video_recorder.reset()  # 비디오 레코더 상태 초기화
                    self.send_gui_update('status', ("MONITORING", None))
                    self.send_gui_update('log', "Movement detected - Returning to monitoring")
                    self.led_controller.led_off()
                    self.send_gui_update('led_status', ("Connected", "LED OFF"))
                    return False

            if is_fall and not self.fall_detection_timestamp:
                self.fall_detection_timestamp = current_time
                self.fall_track_id = track_id
                track['verifying_start_time'] = current_time
                self.send_gui_update('status', ("VERIFYING", self.fall_confirmation_time))
                self.send_gui_update('log', "Fall detected, verifying...")
            
            elif self.fall_detection_timestamp and track_id == self.fall_track_id:
                elapsed_time = current_time - self.fall_detection_timestamp
                
                # detect_fall 메서드 내부의 낙상 확인 부분
                if elapsed_time >= self.fall_confirmation_time:
                    if not self.confirmed_fall:
                        self.confirmed_fall = True
                        self.fall_detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.send_gui_update('status', ("FALL_DETECTED", self.fall_detection_time))
                        self.send_gui_update('log', f"Fall confirmed at {self.fall_detection_time}")
                        self.led_controller.led_on()
                        self.send_gui_update('led_status', ("Connected", "LED ON"))
                        
                        # 서버 통신을 별도 스레드로 실행
                        report_thread = ReportThread(self.report_client, self.arduino_comm, self.gui_queue)
                        report_thread.start()
                        
#                        try:
#                            report_result = self.report_client.send_fall_report()
#                            if report_result and report_result.get('password'):
#                                self.arduino_comm.send_password(report_result['password'])
#                                self.send_gui_update('log', f"Report sent and password transferred to Arduino")
#                        except Exception as e:
#                            self.send_gui_update('log', f"Error in fall reporting: {str(e)}")
                else:
                    remaining_time = self.fall_confirmation_time - elapsed_time
                    self.send_gui_update('status', ("VERIFYING", remaining_time))
            
            self.send_gui_update('metrics', (self.fall_score, len(self.tracks)))

            return self.confirmed_fall and track_id == self.fall_track_id

        except Exception as e:
            self.send_gui_update('log', f"Error in fall detection: {str(e)}")
            return False

def app_callback(pad, info, user_data):
    print("[DEBUG] Starting frame processing")
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    try:
        format, width, height = get_caps_from_pad(pad)
        frame = get_numpy_from_buffer(buffer, format, width, height)

        if frame is not None:
            fall_status = user_data.confirmed_fall
            is_verifying = user_data.fall_detection_timestamp is not None
            print(f"[DEBUG] Processing frame - fall_status: {fall_status}, is_verifying: {is_verifying}")
            user_data.video_recorder.process_frame(frame, fall_status, is_verifying)

        user_data.increment()
        
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
                    user_data.detect_fall(points, bbox, width, height, track_id)

        print("[DEBUG] Frame processed successfully")
    except Exception as e:
        print(f"[DEBUG] Error in frame processing: {str(e)}")

    return Gst.PadProbeReturn.OK
    
def main():
    """메인 함수"""
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
        if hasattr(user_data, 'video_recorder'):
            user_data.video_recorder.cleanup()
        if hasattr(user_data, 'ptz_tracker'):
            user_data.ptz_tracker.cleanup()
        if hasattr(user_data, 'report_client'):
            user_data.report_client.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    app.run()

if __name__ == "__main__":
    main()