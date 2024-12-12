import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
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
        self.confidence_threshold = 0.4
        
        # Initialize PTZ tracking
        self.ptz_tracker = PTZTracker()
        Thread(target=self.ptz_tracker.start_server).start()

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        if label == "person" and confidence > user_data.confidence_threshold:
            # PTZ tracking
            x_movement, y_movement = user_data.ptz_tracker.calculate_movement(bbox, width, height)
            if x_movement != 0 or y_movement != 0:
                user_data.ptz_tracker.send_movement_command(x_movement, y_movement)

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
        if hasattr(user_data, 'ptz_tracker'):
            user_data.ptz_tracker.cleanup()
