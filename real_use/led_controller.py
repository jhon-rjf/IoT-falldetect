import requests
import time
from phue import Bridge

class LEDController:
    def __init__(self, esp_ip, hue_bridge_ip="192.168.0.4"):
        """LED 컨트롤러 및 Hue Bridge 초기화
        esp_ip: ESP8266의 IP 주소
        hue_bridge_ip: Philips Hue Bridge의 IP 주소"""
        print(f"[DEBUG] Initializing LED Controller with IP: {esp_ip}")
        self.base_url = f"http://{esp_ip}"
        self.led_state = False
        
        # Hue Bridge 초기화
        print(f"[DEBUG] Initializing Hue Bridge with IP: {hue_bridge_ip}")
        self.hue_bridge = None
        self.hue_lights = None
        try:
            print("[DEBUG] Attempting to connect to Hue Bridge...")
            self.hue_bridge = Bridge(hue_bridge_ip)
            self.hue_bridge.connect()
            self.hue_lights = self.hue_bridge.get_light_objects("id")
            print(f"[DEBUG] Successfully connected to Hue Bridge. Found {len(self.hue_lights)} lights")
        except Exception as e:
            print(f"[DEBUG] Initial Hue Bridge connection failed: {str(e)}")
            print("[DEBUG] Please press the button on the Hue Bridge within 30 seconds")
            try:
                time.sleep(30)  # 사용자가 버튼을 누를 시간
                self.hue_bridge = Bridge(hue_bridge_ip)
                self.hue_bridge.connect()
                self.hue_lights = self.hue_bridge.get_light_objects("id")
                print("[DEBUG] Successfully connected to Hue Bridge after button press")
            except Exception as e:
                print(f"[DEBUG] Final Hue Bridge connection attempt failed: {str(e)}")
                self.hue_bridge = None
                self.hue_lights = None
            
    def led_on(self):
        """LED 스트립과 Hue 조명 켜기"""
        # LED 스트립 켜기
        if not self.led_state:
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
        
        # Hue 조명 켜기
        if self.hue_bridge and self.hue_lights:
            try:
                print("[DEBUG] Turning on Hue lights")
                for light in self.hue_lights.values():
                    light.on = True
                    light.brightness = 254  # 최대 밝기
                print("[DEBUG] Hue lights turned ON")
            except Exception as e:
                print(f"[DEBUG] Error turning on Hue lights: {str(e)}")
            
    def led_off(self):
        """LED 스트립과 Hue 조명 끄기"""
        # LED 스트립 끄기
        if self.led_state:
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
        
        # Hue 조명 끄기
        if self.hue_bridge and self.hue_lights:
            try:
                print("[DEBUG] Turning off Hue lights")
                for light in self.hue_lights.values():
                    light.on = False
                print("[DEBUG] Hue lights turned OFF")
            except Exception as e:
                print(f"[DEBUG] Error turning off Hue lights: {str(e)}")
