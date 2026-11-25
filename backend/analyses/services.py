import os
import requests
from django.conf import settings

def call_fastapi_inference(video_path: str) -> dict:
    """
    FastAPI AI 추론 서비스(포트 8001)의 /predict/video 에
    영상 파일을 multipart/form-data로 전송하고 결과를 반환합니다.
    """
    ai_base_url = getattr(settings, "AI_SERVICE_URL", "http://localhost:8001").rstrip("/")
    url = f"{ai_base_url}/predict/video"

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"분석할 영상 파일을 찾을 수 없습니다: {video_path}")

    filename = os.path.basename(video_path)
    with open(video_path, "rb") as f:
        files = {"file": (filename, f, "video/mp4")}
        response = requests.post(url, files=files, timeout=180)

    if response.status_code != 200:
        raise RuntimeError(f"FastAPI AI 서버 오류 (HTTP {response.status_code}): {response.text}")

    return response.json()
