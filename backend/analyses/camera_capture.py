import os
import time
import uuid
import cv2
from pathlib import Path
from django.conf import settings

def capture_ip_webcam_video(camera_url: str, duration_sec: int = 8, target_fps: int = 25) -> str:
    """
    IP Webcam 스트림 URL에서 지정된 시간(초) 동안 프레임을 수집하여 MP4 파일로 저장합니다.
    - BUFFERSIZE 최적화로 지연 제거
    - 실제 수신 FPS 동적 계산으로 부드러운 재생 보장
    - H.264 웹 표준 트랜스코딩
    """
    if not camera_url:
        raise ValueError("카메라 스트림 URL이 제공되지 않았습니다.")

    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        raise ConnectionError(f"카메라 스트림에 연결할 수 없습니다: {camera_url}")

    # 버퍼 지연 최소화 (네트워크 버퍼링 방지)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    output_dir = Path(settings.MEDIA_ROOT) / "camera_captures"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"capture_{uuid.uuid4().hex[:8]}.mp4"
    output_path = output_dir / filename

    captured_frames = []
    start_time = time.time()

    try:
        while (time.time() - start_time) < duration_sec:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)  # 5ms 최소 대기
                continue
            captured_frames.append(frame)
    finally:
        cap.release()

    total_frames = len(captured_frames)
    elapsed_time = max(0.1, time.time() - start_time)

    if total_frames < 15:
        raise RuntimeError(f"유효한 영상 프레임 수가 너무 적습니다 ({total_frames}개 프레임). 스마트폰 카메라 연결 및 Wi-Fi 상태를 확인하세요.")

    # 실제 수신된 속도에 맞추어 정확한 FPS 자동 계산 (끊김 없이 부드럽게 재생)
    actual_fps = round(total_frames / elapsed_time, 2)
    actual_fps = max(10.0, min(60.0, actual_fps))

    # 비디오 파일 저장
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, actual_fps, (width, height))
    for f in captured_frames:
        out.write(f)
    out.release()

    # 브라우저 재생을 위한 H.264 웹 표준 인코딩
    import subprocess
    h264_path = str(output_path) + ".h264.mp4"
    try:
        ret = subprocess.run([
            "ffmpeg", "-y", "-i", str(output_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            h264_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if ret.returncode == 0 and os.path.exists(h264_path):
            os.replace(h264_path, str(output_path))
    except Exception as e:
        pass

    return str(output_path)
