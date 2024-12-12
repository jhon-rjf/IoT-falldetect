

from tkinter import Tk, Label, Button, Scale, HORIZONTAL
from phue import Bridge

# 1. Hue Bridge 연결
def connect_to_hue_bridge():
    bridge_ip = "192.168.0.4"  # Hue Bridge의 IP
    try:
        bridge = Bridge(bridge_ip)
        bridge.connect()  # 인증 (Bridge 버튼 눌러야 함)
        print(f"Hue Bridge connection seccess: {bridge_ip}")
        return bridge
    except Exception as e:
        print("Hue Bridge connection failed:", e)
        return None

# 2. 전구 상태 업데이트
def update_lights():
    global lights
    lights = bridge.get_light_objects("id")
    print("status of connected balb:")
    for light_id, light in lights.items():
        print(f"ID: {light_id}, name: {light.name}, status: {'on' if light.on else 'off'}")

# 3. 전구 제어 함수
def toggle_light(light_id):
    light = lights[light_id]
    light.on = not light.on
    update_gui()

def change_brightness(light_id, brightness):
    light = lights[light_id]
    light.brightness = brightness

# 4. GUI 업데이트
def update_gui():
    for light_id, light in lights.items():
        light_buttons[light_id]["text"] = f"{light.name} {'on' if light.on else 'off'}"

# 5. GUI 설정
def create_gui():
    global light_buttons
    light_buttons = {}
    
    root = Tk()
    root.title("Hue balb contoll GUI")

    Label(root, text="Hue light control", font=("Arial", 16)).grid(row=0, columnspan=3, pady=10)

    row = 1
    for light_id, light in lights.items():
        # 전구 이름 및 상태 버튼
        btn = Button(root, text=f"{light.name} {'on' if light.on else 'off'}", 
                     command=lambda lid=light_id: toggle_light(lid), width=20)
        btn.grid(row=row, column=0, padx=10, pady=5)
        light_buttons[light_id] = btn

        # 밝기 조정 슬라이더
        slider = Scale(root, from_=1, to=254, orient=HORIZONTAL, 
                       command=lambda val, lid=light_id: change_brightness(lid, int(val)))
        slider.set(light.brightness)
        slider.grid(row=row, column=1, padx=10, pady=5)

        row += 1

    root.mainloop()

# 메인 실행
if __name__ == "__main__":
    bridge = connect_to_hue_bridge()
    if bridge:
        update_lights()
        create_gui()
