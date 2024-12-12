import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
    get_default_parser,
    QUEUE,
    INFERENCE_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
)
from detection_pipeline import GStreamerDetectionApp

class CustomGStreamerDetectionApp(GStreamerDetectionApp):
    def get_pipeline_string(self):
        # PiCamera source pipeline
        source_pipeline = (
            f'libcamerasrc ! '
            f'video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! '
            f'{QUEUE(name="source_convert_q")} ! '
            f'videoconvert n-threads=3 name=source_convert qos=false ! '
            f'video/x-raw,format=RGB,pixel-aspect-ratio=1/1'
        )
        
        # Get the rest of the pipeline from parent class configuration
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            batch_size=self.batch_size,
            additional_params=self.thresholds_str)
            
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)
        
        # Combine all pipeline parts
        pipeline_string = (
            f'{source_pipeline} ! '
            f'{detection_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )
        
        print(pipeline_string)
        return pipeline_string

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

    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()
        if label == "person":
            string_to_print += f"Detection: {label} {confidence:.2f}\n"
            detection_count += 1
            
    if user_data.use_frame and frame is not None:
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Get the default parser
    parser = get_default_parser()
    args = parser.parse_args()

    # Create an instance of the user app callback class
    user_data = app_callback_class()
    
    # Create and run the detection app with custom pipeline
    app = CustomGStreamerDetectionApp(app_callback, user_data)
    app.run()
