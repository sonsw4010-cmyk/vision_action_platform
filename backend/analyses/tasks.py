import logging
from celery import shared_task
from django.utils import timezone
from .models import AnalysisJob, PersonPrediction
from .services import call_fastapi_inference

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def process_analysis_job(self, job_id: int):
    """
    Celery 비동기 분석 워커 함수
    1. 작업 상태를 PROCESSING으로 변경
    2. FastAPI AI 서비스 호출
    3. PersonPrediction ORM 생성 및 저장
    4. 스켈레톤 분석 영상(Annotated Video) 생성 및 저장
    5. 작업 상태를 SUCCESS (실패 시 FAILURE)로 완료
    """
    try:
        job = AnalysisJob.objects.get(id=job_id)
    except AnalysisJob.DoesNotExist:
        logger.error(f"AnalysisJob #{job_id}를 찾을 수 없습니다.")
        return {"error": f"Job #{job_id} not found"}

    task_id = self.request.id or "sync-test-task"
    job.status = "PROCESSING"
    job.task_id = task_id
    job.save(update_fields=["status", "task_id"])
    logger.info(f"[Job #{job_id}] 비동기 분석 시작 (Task: {task_id})")

    try:
        if job.input_type == "upload":
            if not job.video_file or not job.video_file.path:
                raise ValueError("업로드된 영상 파일이 없습니다.")
            target_path = job.video_file.path
        elif job.input_type == "camera":
            from pathlib import Path
            if job.video_file and Path(job.video_file.path).exists():
                target_path = job.video_file.path
            else:
                from .camera_capture import capture_ip_webcam_video
                target_path = capture_ip_webcam_video(job.camera_url, duration_sec=8)
                job.video_file.name = f"camera_captures/{Path(target_path).name}"
        else:
            raise ValueError(f"지원하지 않는 입력 방식입니다: {job.input_type}")

        # FastAPI 추론 요청
        result = call_fastapi_inference(target_path)

        predictions_data = result.get("predictions", [])
        if not predictions_data:
            warning_msg = result.get("message", "영상에서 감지된 사람이 없습니다.")
            job.status = "SUCCESS"
            job.error_message = warning_msg
            job.completed_at = timezone.now()
            job.save(update_fields=["status", "error_message", "completed_at", "video_file"])
            return {"job_id": job_id, "predictions_count": 0, "status": "SUCCESS"}

        # 사람(Track ID)별 추론 결과 ORM 생성
        created_count = 0
        predictions_map = {}
        for pred in predictions_data:
            act_label = pred.get("action_label") or pred.get("predicted_action") or "unknown"
            conf = float(pred.get("confidence", 0.0))
            track_id = pred.get("track_id", 1)
            PersonPrediction.objects.create(
                job=job,
                track_id=track_id,
                action_label=act_label,
                confidence=conf,
                valid_frames=int(pred.get("valid_frames", 0)),
            )
            created_count += 1
            predictions_map[track_id] = f"{act_label} ({conf*100:.1f}%)"

        # 스켈레톤 분석 영상 생성 (Annotated Video with Pose & Labels)
        try:
            from .annotator import create_annotated_video
            from pathlib import Path
            from django.conf import settings
            import uuid

            annotated_dir = Path(settings.MEDIA_ROOT) / "annotated_videos"
            annotated_dir.mkdir(parents=True, exist_ok=True)
            annotated_filename = f"annotated_{uuid.uuid4().hex[:8]}.mp4"
            annotated_full_path = str(annotated_dir / annotated_filename)

            create_annotated_video(target_path, annotated_full_path, predictions_map)
            job.annotated_video.name = f"annotated_videos/{annotated_filename}"
            logger.info(f"[Job #{job_id}] 스켈레톤 분석 영상 생성 완료: {job.annotated_video.name}")
        except Exception as anno_err:
            logger.warning(f"[Job #{job_id}] 스켈레톤 영상 생성 중 오류: {anno_err}")

        job.status = "SUCCESS"
        job.error_message = None
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "completed_at", "video_file", "annotated_video"])
        logger.info(f"[Job #{job_id}] 분석 완료: {created_count}명 결과 저장됨.")
        return {"job_id": job_id, "predictions_count": created_count, "status": "SUCCESS"}

    except Exception as exc:
        logger.exception(f"[Job #{job_id}] 분석 실패: {exc}")
        job.status = "FAILURE"
        job.error_message = str(exc)
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "completed_at"])
        raise exc
