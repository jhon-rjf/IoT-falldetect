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
        
        # Create pipeline with post-processing elements
        pipeline_str = (
            f"appsrc name=source ! "
            f"videoconvert ! "
            f"video/x-raw,format=RGB ! "
            f"tee name=t ! "
            f"queue ! "
            f"hailonet config-path=/usr/local/hailo/detection.hef ! "
            f"queue ! "
            f"hailofilter function-name=filter_by_score min-score=0.4 ! "
            f"queue ! "
            f"hailotracker name=hailo_tracker tracking-type=bytetrack iou-threshold=0.5 ! "
            f"queue ! "
            f"hailooverlay ! "
            f"queue ! "
            f"videoconvert ! "
            f"autovideosink "
            f"t. ! "
            f"queue ! "
            f"fakesink name=hailo_sink"
        )
        
        self.pipeline = Gst.parse_launch(pipeline_str)
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
        
        # Set pipeline state to playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Failed to set pipeline to PLAYING")
            return
        
        try:
            # Create user callback instance
            user_data = user_app_callback_class()
            
            # Add probe to get detections
            hailo_sink = self.pipeline.get_by_name('hailo_sink')
            pad = hailo_sink.get_static_pad('sink')
            pad.add_probe(
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
            print("\nStopping...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.picam.stop()

if __name__ == "__main__":
    detection = PiCameraDetection()
    detection.run()
