import requests
import time

class LEDController:
    def __init__(self, esp_ip):  # 언더스코어 2개로 수정
        """LED 컨트롤러 초기화
        esp_ip: ESP8266의 IP 주소"""
        print(f"[DEBUG] Initializing LED Controller with IP: {esp_ip}")
        self.base_url = f"http://{esp_ip}"
        self.led_state = False
        
    def led_on(self):
        """LED 켜기"""
        if not self.led_state:  # LED가 꺼져있을 때만 켜기 요청
            try:
                print("[DEBUG] Sending LED ON request")
                response = requests.get(f"{self.base_url}/ledon")
                if response.status_code == 200:
                    print("[DEBUG] Fall Alert: LED ON")
                    self.led_state = True
                else:
                    print(f"[DEBUG] LED ON failed: status code {response.status_code}")
            except Exception as e:
                print(f"[DEBUG] Error in LED ON: {str(e)}")
            
    def led_off(self):
        """LED 끄기"""
        if self.led_state:  # LED가 켜져있을 때만 끄기 요청
            try:
                print("[DEBUG] Sending LED OFF request")
                response = requests.get(f"{self.base_url}/ledoff")
                if response.status_code == 200:
                    print("[DEBUG] LED OFF")
                    self.led_state = False
                else:
                    print(f"[DEBUG] LED OFF failed: status code {response.status_code}")
            except Exception as e:
                print(f"[DEBUG] Error in LED OFF: {str(e)}")