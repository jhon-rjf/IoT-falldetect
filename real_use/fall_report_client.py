import requests
import random
import string
import time
from datetime import datetime
from config import Config
import serial

class FallReportClient:
    def __init__(self):
        self.config = Config()
        self.last_report_time = None
        self.session = requests.Session()
        self.bluetooth = None

        # Bluetooth 설정
        if self.config.ENABLE_BLUETOOTH:
            self.bluetooth = serial.Serial(
                port=self.config.BLUETOOTH_PORT,
                baudrate=self.config.BLUETOOTH_BAUD_RATE
            )
    
    def __del__(self):
        """소멸자에서 세션과 블루투스 정리"""
        self.session.close()
        if self.bluetooth:
            self.bluetooth.close()

    def generate_password(self):
        """랜덤 비밀번호 생성 (숫자로만, 4자리)"""
        return ''.join(random.choice(string.digits) for _ in range(self.config.PASSWORD_LENGTH))

    def can_send_report(self):
        """보고서 전송 가능 여부 확인 (도배 방지)"""
        if self.last_report_time is None:
            return True
        time_since_last_report = time.time() - self.last_report_time
        return time_since_last_report >= self.config.MIN_REPORT_INTERVAL

    def send_password_to_arduino(self, password):
        """비밀번호를 아두이노로 송신"""
        if not self.bluetooth:
            print("Bluetooth not enabled or unavailable")
            return False
        
        try:
            print(f"Sending password to Arduino: {password}")
            self.bluetooth.write((password + '\n').encode())
            time.sleep(1)  # 데이터 송신 후 약간의 지연
            print("Password sent successfully")
            return True
        except Exception as e:
            print(f"Error sending password to Arduino: {e}")
            return False

    def send_fall_report(self):
        """낙상 신고 전송"""
        if not self.can_send_report():
            print("Report blocked: Too soon since last report")
            return None

        # 신고 데이터 준비
        password = self.generate_password()
        report_data = {
            "address": self.config.FIXED_ADDRESS,
            "password": password,
            "name": self.config.FIXED_NAME,
            "date": datetime.now().isoformat()
        }

        # 재시도 로직
        for attempt in range(self.config.REPORT_RETRY_COUNT):
            try:
                with self.session.post(
                    f"{self.config.SERVER_URL}/reporting",
                    json=report_data,
                    timeout=5
                ) as response:
                    if response.status_code == 200:
                        print(f"Report sent successfully with password: {password}")
                        self.last_report_time = time.time()

                        # 비밀번호 아두이노로 송신
                        self.send_password_to_arduino(password)

                        response.close()
                        return {"success": True, "password": password}
                    else:
                        print(f"Server returned status code: {response.status_code}")
                        response.close()

            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.REPORT_RETRY_COUNT - 1:
                    time.sleep(self.config.REPORT_RETRY_DELAY)
                continue

        print("Failed to send report after all attempts")
        return None

    def cleanup(self):
        """리소스 정리"""
        self.session.close()
        if self.bluetooth:
            self.bluetooth.close()
