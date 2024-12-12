import serial
import time

port = "/dev/rfcomm0"
baud_rate = 9600

bluetooth = serial.Serial(port, baud_rate)

password = "1333"  # 전송할 비밀번호, 외부 혹은 웹에서 끌어와야 하는 값이고, 정수형이 아닌 문자열이어야 함

try:
    print("Sending password:", password)
    bluetooth.write((password + '\n').encode())  # '\n'으로 데이터 끝 구분
    time.sleep(1)

except Exception as e:
    print("Error:", e)

finally:
    bluetooth.close()
    print("Disconnected")
