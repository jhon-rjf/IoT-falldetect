import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
import hailo
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from detection_pipeline import GStreamerDetectionApp
from picamera2 import Picamera2

class PiCameraDetection:
    def __init__(self, width=640, height=360):
        self.width = width
        self.height = height
        
        # Initialize PiCamera2
        self.picam = Picamera2()
        self.picam.configure(self.picam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameDurationLimits": (33333, 33333)}  # 30fps
        ))
        
        # Create custom GStreamer pipeline
        pipeline_str = (
            f"appsrc name=source ! "
            f"video/x-raw,format=RGB,width={self.width},height={self.height},framerate=30/1 ! "
            f"videoconvert ! "
            f"video/x-raw,format=RGB ! "
            f"tee name=t ! "
            f"queue ! "
            f"videoconvert ! "
            f"hailo_net config-path=/usr/local/hailo/detection.hef ! "
            f"queue ! "
            f"fakesink name=hailo_sink t. ! "
            f"queue ! "
            f"videoconvert ! "
            f"autovideosink"
        )
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name('source')
        
        # Setup appsrc
        self.appsrc.set_property('format', Gst.Format.TIME)
        self.appsrc.set_property('block', True)
        
    def frame_callback(self):
        while True:
            # Capture frame from PiCamera2
            frame = self.picam.capture_array()
            
            # Convert frame to GStreamer buffer
            buffer = Gst.Buffer.new_wrapped(frame.tobytes())
            
            # Push buffer to pipeline
            self.appsrc.emit('push-buffer', buffer)
            
    def run(self):
        # Start PiCamera2
        self.picam.start()
        
        # Start GStreamer pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        
        try:
            # Create user callback instance
            user_data = user_app_callback_class()
            
            # Add probe to get detections
            hailo_sink = self.pipeline.get_by_name('hailo_sink')
            hailo_sink.get_static_pad('sink').add_probe(
                Gst.PadProbeType.BUFFER,
                app_callback,
                user_data
            )
            
            # Start frame capture thread
            import threading
            capture_thread = threading.Thread(target=self.frame_callback)
            capture_thread.daemon = True
            capture_thread.start()
            
            # Run main loop
            loop = GLib.MainLoop()
            loop.run()
            
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
            
    def cleanup(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.picam.stop()

if __name__ == "__main__":
    # Initialize GStreamer
    Gst.init(None)
    
    # Create and run detection pipeline
    detection = PiCameraDetection()
    detection.run()
