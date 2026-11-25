import time
import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException

try:
    from ..services.pose_pipeline import pose_pipeline
    from ..services.action_inference import action_inference
except ImportError:
    from services.pose_pipeline import pose_pipeline
    from services.action_inference import action_inference

router = APIRouter(tags=["Predictions"])

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@router.post("/predict/video")
@router.post("/predict/")
async def predict_video(file: UploadFile = File(...)):
    """
    영상 파일을 입력받아 사람별 동작을 예측하고 신뢰도와 함께 반환합니다.
    (교안 규격: 6개 동작 모델 기준)
    """
    start_time = time.time()
    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 영상 파일 형식입니다: {ext} (지원 형식: {ALLOWED_EXTENSIONS})",
        )

    # 1. 업로드된 영상을 임시 파일에 안전하게 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_path = Path(temp_file.name)
        shutil.copyfileobj(file.file, temp_file)

    try:
        # 2. YOLO Pose로 사람별 1,140차원 시퀀스 추출
        sequences = pose_pipeline.extract_person_sequences(temp_path)
        if not sequences:
            raise HTTPException(
                status_code=400,
                detail="영상에서 유효한 사람의 동작을 감지하지 못했습니다.",
            )

        # 3. 6개 Action Random Forest 모델로 동작 분류
        predictions = action_inference.infer_actions(sequences)

        elapsed = round(time.time() - start_time, 2)

        return {
            "status": "success",
            "filename": file.filename,
            "person_count": len(predictions),
            "predictions": predictions,
            "processing_time": elapsed,
        }
    finally:
        # 4. 분석 완료 후 임시 파일 즉시 삭제
        if temp_path.exists():
            temp_path.unlink()
