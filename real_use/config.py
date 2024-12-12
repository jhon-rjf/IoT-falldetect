class Config:
    # 서버 설정
    SERVER_HOST = "192.168.0.3"  # 실제 서버 IP로 변경 필요
    SERVER_PORT = 8000
    SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
    
    # 신고 데이터 설정
    FIXED_ADDRESS = "안동대학교 공학 1호관 4131호"
    FIXED_NAME = "홍김동동"
    
    # 아두이노 설정
    ARDUINO_PORT = "/dev/ttyUSB0"  # Windows의 경우 "COM3" 등으로 변경
    ARDUINO_BAUDRATE = 9600
    ARDUINO_TIMEOUT = 1
    
    # 비밀번호 설정
    PASSWORD_LENGTH = 4
    PASSWORD_RETRY_COUNT = 3
    
    # 신고 관련 설정
    REPORT_RETRY_COUNT = 3
    REPORT_RETRY_DELAY = 2  # seconds
    MIN_REPORT_INTERVAL = 100  # seconds (5 minutes)
    
    # Bluetooth 관련 설정 추가
    ENABLE_BLUETOOTH = True
    BLUETOOTH_PORT = "/dev/rfcomm0"  # 포트 이름
    BLUETOOTH_BAUD_RATE = 9600       # 보드 속도
