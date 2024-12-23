import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
import time

class FallDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fall Detection System')
        self.setMinimumSize(800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Status Section
        self.status_frame = self.create_status_section()
        layout.addWidget(self.status_frame)
        
        # Info Section
        info_layout = QHBoxLayout()
        # Left Column (Metrics)
        self.metrics_frame = self.create_metrics_section()
        info_layout.addWidget(self.metrics_frame)
        
        # Right Column (LED & PTZ Status)
        status_layout = QVBoxLayout()
        self.led_frame = self.create_led_section()
        self.ptz_frame = self.create_ptz_section()
        status_layout.addWidget(self.led_frame)
        status_layout.addWidget(self.ptz_frame)
        
        right_widget = QWidget()
        right_widget.setLayout(status_layout)
        info_layout.addWidget(right_widget)
        
        layout.addLayout(info_layout)
        
        # Event Log Section
        self.log_frame = self.create_log_section()
        layout.addWidget(self.log_frame)
        
        # Initialize variables
        self.fall_detection_time = None
        self.fall_score = 0
        self.active_tracks = 0
        self.led_status = "Initializing..."
        self.ptz_status = "Waiting for connection..."
        self.last_led_command = None
        
        # Style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
            QFrame {
                border-radius: 10px;
                background-color: white;
                border: 1px solid #e1e8ed;
            }
            QLabel {
                color: #2c3e50;
            }
        """)

    def create_status_section(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        
        self.status_label = QLabel("MONITORING")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.status_label.setStyleSheet("color: #27ae60;")
        
        self.time_label = QLabel("")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setFont(QFont("Arial", 12))
        
        layout.addWidget(self.status_label)
        layout.addWidget(self.time_label)
        
        return frame

    def create_metrics_section(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        
        title = QLabel("Metrics")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        
        self.fall_score_label = QLabel("Fall Score: 0.0")
        self.fall_score_label.setFont(QFont("Arial", 12))
        
        self.tracks_label = QLabel("Active Tracks: 0")
        self.tracks_label.setFont(QFont("Arial", 12))
        
        layout.addWidget(title)
        layout.addWidget(self.fall_score_label)
        layout.addWidget(self.tracks_label)
        
        return frame

    def create_led_section(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        
        title = QLabel("LED Control")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        
        self.led_status_label = QLabel("Status: Initializing...")
        self.led_status_label.setFont(QFont("Arial", 12))
        
        self.led_command_label = QLabel("Last Command: None")
        self.led_command_label.setFont(QFont("Arial", 12))
        
        layout.addWidget(title)
        layout.addWidget(self.led_status_label)
        layout.addWidget(self.led_command_label)
        
        return frame

    def create_ptz_section(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        
        title = QLabel("PTZ Status")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        
        self.ptz_status_label = QLabel("Status: Waiting for connection...")
        self.ptz_status_label.setFont(QFont("Arial", 12))
        
        layout.addWidget(title)
        layout.addWidget(self.ptz_status_label)
        
        return frame

    def create_log_section(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        
        title = QLabel("Event Log")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        
        self.log_label = QLabel("System initialized")
        self.log_label.setFont(QFont("Arial", 12))
        self.log_label.setWordWrap(True)
        
        layout.addWidget(title)
        layout.addWidget(self.log_label)
        
        return frame

    def update_status(self, status, detection_time=None):
        if status == "MONITORING":
            self.status_label.setText("MONITORING")
            self.status_label.setStyleSheet("color: #27ae60;")
            self.time_label.setText("")
        elif status == "VERIFYING":
            self.status_label.setText("Fall Detected, Verifying...")
            self.status_label.setStyleSheet("color: #f39c12;")
            if detection_time is not None:
                self.time_label.setText(f"Verifying... ({detection_time:.1f}s)")
        elif status == "FALL_DETECTED":
            self.status_label.setText("FALL DETECTED")
            self.status_label.setStyleSheet("color: #c0392b;")
            if detection_time:
                self.time_label.setText(f"Detected at: {detection_time}")

    def update_metrics(self, fall_score, active_tracks):
        self.fall_score_label.setText(f"Fall Score: {fall_score:.1f}")
        self.tracks_label.setText(f"Active Tracks: {active_tracks}")

    def update_led_status(self, status, last_command):
        self.led_status_label.setText(f"Status: {status}")
        self.led_command_label.setText(f"Last Command: {last_command}")

    def update_ptz_status(self, status):
        self.ptz_status_label.setText(f"Status: {status}")

    def add_log_message(self, message):
        self.log_label.setText(message)

def create_gui():
    return FallDetectionGUI()
