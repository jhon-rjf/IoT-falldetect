from detection_pipeline import GStreamerDetectionApp
from picamera2 import Picamera2
import numpy as np
import cv2
import os

class PiCameraSource:
    def __init__(self, width=640, height=360):
        self.picam = Picamera2()
        self.picam.configure(self.picam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        ))
        self.picam.start()

    def read(self):
        return True, self.picam.capture_array()

    def release(self):
        self.picam.stop()

if __name__ == "__main__":
    # Create camera source
    camera = PiCameraSource()
    
    # Pass camera source to GStreamerDetectionApp
    app = GStreamerDetectionApp(camera=camera)
    app.run()
