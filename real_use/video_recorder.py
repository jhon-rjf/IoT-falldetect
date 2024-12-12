import cv2
import numpy as np
from collections import deque
from threading import Thread, Lock
from datetime import datetime
import os
import time
import psutil

class VideoBuffer:
    def __init__(self, buffer_seconds=5, fps=30):
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.frame_buffer = deque(maxlen=buffer_seconds * fps)
        self.lock = Lock()
        self.is_recording = False
        self.current_video = None
        self.video_path = "fall_videos"
        self.recording_completed = False
        self.frame_count = 0
        self.last_cleanup_time = time.time()
        
        if not os.path.exists(self.video_path):
            os.makedirs(self.video_path)
    
    def add_frame(self, frame):
        """프레임을 버퍼에 추가"""
        try:
            with self.lock:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_buffer.append(rgb_frame.copy())
                self.frame_count += 1

                # 주기적으로 버퍼 상태 체크 및 정리
                current_time = time.time()
                if current_time - self.last_cleanup_time > 5:  # 5초마다 체크
                    print(f"[DEBUG] Buffer status - Frames: {self.frame_count}, Buffer size: {len(self.frame_buffer)}")
                    self.last_cleanup_time = current_time
                    
                    # 버퍼가 너무 커지면 정리
                    if len(self.frame_buffer) > self.fps * self.buffer_seconds:
                        print("[DEBUG] Cleaning buffer")
                        while len(self.frame_buffer) > self.fps * self.buffer_seconds:
                            self.frame_buffer.popleft()

        except Exception as e:
            print(f"[DEBUG] Error in add_frame: {str(e)}")
    
    def start_recording(self):
        """녹화 시작"""
        if self.is_recording or self.recording_completed:
            return
            
        with self.lock:
            self.is_recording = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = f"{self.video_path}/fall_{timestamp}.mp4"
            
            if len(self.frame_buffer) > 0:
                height, width = self.frame_buffer[0].shape[:2]
                self.current_video = cv2.VideoWriter(
                    video_name,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.fps,
                    (width, height)
                )
                
                # 이전 5초 프레임들 저장
                for frame in self.frame_buffer:
                    self.current_video.write(frame)
                print(f"[DEBUG] Started recording to {video_name}")
    
    def stop_recording(self, save=True):
        """녹화 중지"""
        if not self.is_recording:
            return
            
        with self.lock:
            self.is_recording = False
            if self.current_video is not None:
                self.current_video.release()
                
                if not save:
                    video_files = sorted(os.listdir(self.video_path))
                    if video_files:
                        latest_video = os.path.join(self.video_path, video_files[-1])
                        os.remove(latest_video)
                        print(f"[DEBUG] Deleted video: {latest_video}")
                else:
                    self.recording_completed = True
                    print("[DEBUG] Recording saved and completed")
                
                self.current_video = None

    def record_frame(self, frame):
        """현재 프레임을 녹화"""
        if self.is_recording and self.current_video is not None and not self.recording_completed:
            with self.lock:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_video.write(rgb_frame)
                print("[DEBUG] Frame recorded")

    def cleanup(self):
        """리소스 정리"""
        with self.lock:
            if self.current_video is not None:
                self.current_video.release()
            self.frame_buffer.clear()
            self.frame_count = 0
            self.is_recording = False
            print("[DEBUG] VideoBuffer cleanup completed")

    def reset(self):
        """상태 초기화"""
        with self.lock:
            self.frame_buffer.clear()
            self.frame_count = 0
            self.recording_completed = False
            self.is_recording = False
            if self.current_video is not None:
                self.current_video.release()
                self.current_video = None
            print("[DEBUG] VideoBuffer reset completed")

class FallVideoRecorder:
    def __init__(self, gui_queue):
        self.gui_queue = gui_queue
        self.video_buffer = VideoBuffer()
        self.post_fall_frames = 0
        self.required_post_frames = 5 * 30  # 5초 * 30fps
        self.last_memory_check = time.time()
    
    def add_frame(self, frame):
        """프레임을 버퍼에 추가"""
        try:
            self.video_buffer.add_frame(frame)
            print("[DEBUG] Frame added to buffer")
        except Exception as e:
            print(f"[DEBUG] Error adding frame to buffer: {str(e)}")

    def process_frame(self, frame, fall_status, is_verifying):
        """프레임 처리 및 녹화 상태 관리"""
        print(f"[DEBUG] Video buffer - fall_status: {fall_status}, is_verifying: {is_verifying}")
        
        try:
            # 메모리 사용량 체크
            current_time = time.time()
            if current_time - self.last_memory_check > 5:
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                print(f"[DEBUG] Memory usage: {memory_usage:.2f} MB")
                self.last_memory_check = current_time

            # 프레임을 버퍼에 추가
            self.add_frame(frame)
            
            # 이미 녹화가 완료된 경우 추가 처리하지 않음
            if self.video_buffer.recording_completed:
                return
                
            # 낙상 감지 중(VERIFYING) 상태이면 녹화 시작
            if is_verifying and not self.video_buffer.is_recording:
                self.video_buffer.start_recording()
                self.gui_queue.put(('log', "Started recording fall video"))
                
            # VERIFYING 상태가 아니고 녹화 중이면 (비낙상으로 판정)
            elif not is_verifying and self.video_buffer.is_recording:
                self.video_buffer.stop_recording(save=False)
                self.gui_queue.put(('log', "Cancelled fall video recording"))
                self.post_fall_frames = 0
                
            # 낙상이 확인되고 녹화 중이면 추가 5초 녹화
            elif fall_status and self.video_buffer.is_recording:
                self.video_buffer.record_frame(frame)
                self.post_fall_frames += 1
                
                # 추가 5초 녹화 완료
                if self.post_fall_frames >= self.required_post_frames:
                    self.video_buffer.stop_recording(save=True)
                    self.gui_queue.put(('log', "Completed fall video recording"))
                    self.post_fall_frames = 0
            
            # 녹화 중이면 현재 프레임 저장
            elif self.video_buffer.is_recording:
                self.video_buffer.record_frame(frame)
                
        except Exception as e:
            print(f"[DEBUG] Error in process_frame: {str(e)}")
            self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'video_buffer') and self.video_buffer.is_recording:
                self.video_buffer.stop_recording(save=True)
            print("[DEBUG] FallVideoRecorder cleanup completed")
        except Exception as e:
            print(f"[DEBUG] Error during cleanup: {str(e)}")

    def reset(self):
        """녹화 상태 초기화"""
        try:
            if hasattr(self, 'video_buffer'):
                self.video_buffer.reset()
            self.post_fall_frames = 0
            print("[DEBUG] FallVideoRecorder reset completed")
        except Exception as e:
            print(f"[DEBUG] Error during reset: {str(e)}")
