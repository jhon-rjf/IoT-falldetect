import socket
import json
import threading

class PTZTracker:
    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', 5000))
        self.server_socket.listen(1)
        self.client = None
        self.is_connected = False
        self.lock = threading.Lock()

    def start_server(self):
        print("Waiting for PTZ camera connection...")
        self.client, addr = self.server_socket.accept()
        self.is_connected = True
        print(f"Connected to PTZ camera at {addr}")
        return True

    def calculate_movement(self, bbox, width, height):
        center_x = int((bbox.xmin() + bbox.xmax()) * width / 2)
        center_y = int((bbox.ymin() + bbox.ymax()) * height / 2)
        
        left_threshold = width / 3
        right_threshold = (width * 2) / 3
        top_threshold = height / 3
        bottom_threshold = (height * 2) / 3
        
        x_movement = 0
        y_movement = 0
        
        if center_x < left_threshold:
            x_movement = 1
        elif center_x > right_threshold:
            x_movement = -1
            
        if center_y < top_threshold:
            y_movement = 1
        elif center_y > bottom_threshold:
            y_movement = -1
            
        return x_movement, y_movement

    def send_movement_command(self, x_movement, y_movement):
        if not self.is_connected or not self.client:
            return False

        with self.lock:
            try:
                command = {
                    'x_movement': x_movement,
                    'y_movement': y_movement
                }
                command_str = json.dumps(command) + '\n'
                self.client.send(command_str.encode())
                return True
            except Exception as e:
                print(f"Failed to send command: {e}")
                self.is_connected = False
                self.client = None
                return False

    def cleanup(self):
        if self.client:
            self.client.close()
        self.server_socket.close()
        self.is_connected = False
