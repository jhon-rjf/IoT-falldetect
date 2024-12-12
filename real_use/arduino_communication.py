# arduino_communication.py
import serial
import time
from config import Config

class ArduinoCommunication:
    def __init__(self):
        self.config = Config()
        self.serial = None
        self.connect()
    
    def connect(self):
        """아두이노와 시리얼 연결 설정"""
        try:
            self.serial = serial.Serial(
                port=self.config.ARDUINO_PORT,
                baudrate=self.config.ARDUINO_BAUDRATE,
                timeout=self.config.ARDUINO_TIMEOUT
            )
            time.sleep(2)  # 아두이노 리셋 대기
            print("Arduino connected successfully")
        except Exception as e:
            print(f"Arduino connection failed: {str(e)}")
            self.serial = None
    
    def send_password(self, password):
        """비밀번호를 아두이노로 전송"""
        if not self.serial:
            print("Arduino not connected")
            return False
            
        try:
            # 비밀번호 앞뒤로 특수문자 추가하여 전송
            message = f"<{password}>\n"
            self.serial.write(message.encode())
            
            # 아두이노로부터 응답 대기
            response = self.serial.readline().decode().strip()
            success = response == "OK"
            print(f"Password sent to Arduino: {'success' if success else 'failed'}")
            return success
        except Exception as e:
            print(f"Failed to send password to Arduino: {str(e)}")
            return False
    
    def cleanup(self):
        """연결 종료"""
        if self.serial:
            self.serial.close()
            print("Arduino connection closed")
