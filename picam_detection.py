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
        # Initialize GStreamer
        Gst.init(None)
        
        self.width = width
        self.height = height
        
        # Initialize PiCamera2
        self.picam = Picamera2()
        self.picam.configure(self.picam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        ))
        
        # Create basic pipeline
        pipeline_str = (
            f"appsrc name=source ! "
            f"videoconvert ! "
            f"video/x-raw,format=RGB ! "
            f"hailonet ! "  # Basic hailonet without properties
            f"videoconvert ! "
            f"autovideosink"
        )
        
        # Create pipeline
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            print(f"Error creating pipeline: {e}")
            raise
            
        # Get pipeline elements
        self.appsrc = self.pipeline.get_by_name('source')
        
        # Setup appsrc caps
        caps = Gst.Caps.from_string(
            f'video/x-raw,format=RGB,width={self.width},height={self.height},framerate=30/1'
        )
        self.appsrc.set_caps(caps)
        
        # Configure appsrc
        self.appsrc.set_property('format', Gst.Format.TIME)
        self.appsrc.set_property('do-timestamp', True)
        
    def frame_callback(self):
        while True:
            try:
                # Capture frame from PiCamera2
                frame = self.picam.capture_array()
                
                # Create GStreamer buffer
                buffer = Gst.Buffer.new_wrapped(frame.tobytes())
                
                # Push buffer to pipeline
                ret = self.appsrc.emit('push-buffer', buffer)
                if ret != Gst.FlowReturn.OK:
                    print(f"Error pushing buffer: {ret}")
                    
            except Exception as e:
                print(f"Error in frame callback: {e}")
                break
                
    def run(self):
        # Start PiCamera2
        self.picam.start()
        print("PiCamera2 started")
        
        # Set pipeline state to playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Failed to set pipeline to PLAYING")
            return
        print("Pipeline is PLAYING")
        
        try:
            # Start frame capture thread
            import threading
            capture_thread = threading.Thread(target=self.frame_callback)
            capture_thread.daemon = True
            capture_thread.start()
            print("Frame capture thread started")
            
            # Run main loop
            loop = GLib.MainLoop()
            loop.run()
            
        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.picam.stop()
        print("Cleanup completed")

if __name__ == "__main__":
    try:
        detection = PiCameraDetection()
        detection.run()
    except Exception as e:
        print(f"Fatal error: {e}")
