import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from collections import deque
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # 낙상 감지를 위한 파라미터
        self.height_ratio_threshold = 0.5
        self.angle_threshold = 60
        self.fall_history = deque(maxlen=5)
        
    def calculate_height_ratio(self, points, bbox):
        """키포인트 높이 비율 계산"""
        head = points[0]  # nose keypoint
        left_ankle = points[15]
        right_ankle = points[16]
        
        # 정규화된 좌표를 실제 좌표로 변환
        head_x = int((head.x() * bbox.width() + bbox.xmin()))
        head_y = int((head.y() * bbox.height() + bbox.ymin()))
        left_ankle_x = int((left_ankle.x() * bbox.width() + bbox.xmin()))
        left_ankle_y = int((left_ankle.y() * bbox.height() + bbox.ymin()))
        right_ankle_x = int((right_ankle.x() * bbox.width() + bbox.xmin()))
        right_ankle_y = int((right_ankle.y() * bbox.height() + bbox.ymin()))
        
        # 발목의 중간점 계산
        ankle_mid_x = (left_ankle_x + right_ankle_x) / 2
        ankle_mid_y = (left_ankle_y + right_ankle_y) / 2
        
        # 수직 거리와 수평 거리 계산
        vertical_dist = abs(head_y - ankle_mid_y)
        horizontal_dist = abs(head_x - ankle_mid_x)
        
        if horizontal_dist == 0:
            return float('inf')
        return vertical_dist / horizontal_dist

    def calculate_body_angle(self, points, bbox):
        """상체 각도 계산"""
        neck = points[1]  # neck keypoint
        left_hip = points[11]
        right_hip = points[12]
        
        # 정규화된 좌표를 실제 좌표로 변환
        neck_x = int((neck.x() * bbox.width() + bbox.xmin()))
        neck_y = int((neck.y() * bbox.height() + bbox.ymin()))
        left_hip_x = int((left_hip.x() * bbox.width() + bbox.xmin()))
        left_hip_y = int((left_hip.y() * bbox.height() + bbox.ymin()))
        right_hip_x = int((right_hip.x() * bbox.width() + bbox.xmin()))
        right_hip_y = int((right_hip.y() * bbox.height() + bbox.ymin()))
        
        hip_mid_x = (left_hip_x + right_hip_x) / 2
        hip_mid_y = (left_hip_y + right_hip_y) / 2
        
        dx = hip_mid_x - neck_x
        dy = hip_mid_y - neck_y
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        return angle

    def detect_fall(self, points, bbox):
        """낙상 감지"""
        height_ratio = self.calculate_height_ratio(points, bbox)
        body_angle = self.calculate_body_angle(points, bbox)
        
        is_fall = (height_ratio < self.height_ratio_threshold or 
                  body_angle > self.angle_threshold)
        
        self.fall_history.append(is_fall)
        return sum(self.fall_history) >= 3

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

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
        
        if label == "person":
            string_to_print += f"Detection: {label} {confidence:.2f}\n"
            
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                
                # 낙상 감지 수행
                if len(points) >= 17 and user_data.detect_fall(points, bbox):
                    if user_data.use_frame:
                        # 바운딩 박스 그리기 (밝은 주황색)
                        x_min = int(bbox.xmin() * width)
                        y_min = int(bbox.ymin() * height)
                        x_max = int(bbox.xmax() * width)
                        y_max = int(bbox.ymax() * height)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                                    (0, 165, 255), 2)  # 밝은 주황색
                        
                        # 텍스트 배경 그리기
                        text = "FALLDOWN DETECT"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        thickness = 2
                        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                        
                        # 텍스트 배경 (밝은 주황색)
                        cv2.rectangle(frame, 
                                    (x_min, y_min - text_size[1] - 10),
                                    (x_min + text_size[0], y_min),
                                    (0, 165, 255), -1)
                        
                        # 텍스트 (검정색)
                        cv2.putText(frame, text, 
                                  (x_min, y_min - 5),
                                  font, font_scale, (0, 0, 0), thickness)
                
                # 키포인트 시각화
                if user_data.use_frame:
                    for point in points:
                        x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                        y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    if user_data.use_frame:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
