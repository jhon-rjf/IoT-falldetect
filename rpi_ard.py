import serial
import time

port = "/dev/rfcomm0"  # 블루투스 포트
baud_rate = 9600  # 통신 속도

try:
    bluetooth = serial.Serial(port, baud_rate)
    
    # 블루투스 연결 상태 확인
    if bluetooth.is_open:
        print(f"Bluetooth connected on port {port}")
    else:
        print(f"Failed to connect to Bluetooth on port {port}")
        exit(1)
    time.sleep(2)
    password = "1333"  # 전송할 비밀번호 (문자열이어야 함)

    print("Sending password:", password)
    bluetooth.write((password + '\n').encode())  # '\n'으로 데이터 끝 구분
    time.sleep(1)

except serial.SerialException as e:
    print(f"Error: {e}")  # 블루투스 연결 오류 처리
except Exception as e:
    print(f"Unexpected error: {e}")  # 기타 예외 처리

finally:
    if bluetooth.is_open:
        bluetooth.close()
        print("Bluetooth disconnected")
