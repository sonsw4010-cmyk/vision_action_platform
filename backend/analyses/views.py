import json
from django.shortcuts import render, get_object_or_404, redirect
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from django.db.models import Count, Avg
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import AnalysisJob, PersonPrediction
from .serializers import AnalysisJobSerializer, PersonPredictionSerializer
from .tasks import process_analysis_job

@login_required
def analysis_index_view(request):
    if request.user.is_staff:
        qs = AnalysisJob.objects.all().prefetch_related("predictions")
    else:
        qs = AnalysisJob.objects.filter(user=request.user).prefetch_related("predictions")

    try:
        per_page = int(request.GET.get("per_page", 15))
    except (TypeError, ValueError):
        per_page = 15
    if per_page not in (10, 15, 30, 50):
        per_page = 15

    paginator = Paginator(qs, per_page)
    page_obj = paginator.get_page(request.GET.get("page", 1))

    cur = page_obj.number
    total_pages = paginator.num_pages
    start_p = max(1, cur - 2)
    end_p = min(total_pages, cur + 2)
    if end_p - start_p < 4:
        if start_p == 1:
            end_p = min(total_pages, start_p + 4)
        elif end_p == total_pages:
            start_p = max(1, end_p - 4)
    page_range = list(range(start_p, end_p + 1))

    return render(request, "analyses/index.html", {
        "jobs": page_obj,
        "page_obj": page_obj,
        "page_range": page_range,
        "per_page": per_page,
        "total_count": paginator.count,
    })

@login_required
def analysis_detail_view(request, pk):
    if request.user.is_staff:
        job = get_object_or_404(AnalysisJob, pk=pk)
    else:
        job = get_object_or_404(AnalysisJob, pk=pk, user=request.user)
    
    predictions = job.predictions.all()
    for p in predictions:
        if p.confidence <= 1.0:
            p.confidence = p.confidence * 100
            
    return render(request, "analyses/detail.html", {"job": job, "predictions": predictions})

@login_required
def dashboard_view(request):
    """
    관리자(is_staff) 전용 모니터링 대시보드
    """
    if not request.user.is_staff:
        return HttpResponseForbidden("관리자만 접근할 수 있는 페이지입니다.")

    total_jobs = AnalysisJob.objects.count()
    success_jobs = AnalysisJob.objects.filter(status="SUCCESS").count()
    failed_jobs = AnalysisJob.objects.filter(status="FAILURE").count()
    success_rate = (success_jobs / total_jobs * 100) if total_jobs > 0 else 0

    total_persons = PersonPrediction.objects.count()
    avg_conf_raw = PersonPrediction.objects.aggregate(avg=Avg("confidence"))["avg"] or 0.0
    avg_confidence = avg_conf_raw * 100 if avg_conf_raw <= 1.0 else avg_conf_raw

    # 동작별 통계 (17개 한글 동작명 매핑)
    from .models import ACTION_KOREAN_DICT

    action_stats_raw = list(
        PersonPrediction.objects.values("action_label")
        .annotate(count=Count("id"))
        .order_by("-count")[:10]
    )
    action_labels = []
    action_counts = []
    action_stats = []
    for item in action_stats_raw:
        raw_key = item["action_label"]
        ko_name = ACTION_KOREAN_DICT.get(raw_key, raw_key)
        action_labels.append(ko_name)
        action_counts.append(item["count"])
        action_stats.append({
            "action_label": ko_name,
            "count": item["count"]
        })

    # 입력 방식별 통계
    input_stats = list(
        AnalysisJob.objects.values("input_type")
        .annotate(count=Count("id"))
    )
    type_name_map = {"upload": "영상 파일 업로드", "camera": "IP Webcam"}
    input_labels = [type_name_map.get(item["input_type"], item["input_type"]) for item in input_stats]
    input_counts = [item["count"] for item in input_stats]

    dash_qs = AnalysisJob.objects.select_related("user").prefetch_related("predictions")
    dash_paginator = Paginator(dash_qs, 15)
    recent_jobs = dash_paginator.get_page(request.GET.get("page", 1))
    d_cur = recent_jobs.number
    d_tot = dash_paginator.num_pages
    d_start = max(1, d_cur - 2)
    d_end = min(d_tot, d_cur + 2)
    dash_page_range = list(range(d_start, d_end + 1))

    context = {
        "total_jobs": total_jobs,
        "success_jobs": success_jobs,
        "failed_jobs": failed_jobs,
        "success_rate": success_rate,
        "total_persons": total_persons,
        "avg_confidence": avg_confidence,
        "action_labels": json.dumps(action_labels),
        "action_counts": json.dumps(action_counts),
        "action_stats": action_stats,
        "input_labels": json.dumps(input_labels),
        "input_counts": json.dumps(input_counts),
        "input_stats": input_stats,
        "recent_jobs": recent_jobs,
        "dash_page_range": dash_page_range,
    }
    return render(request, "analyses/dashboard.html", context)

class AnalysisJobViewSet(viewsets.ModelViewSet):
    serializer_class = AnalysisJobSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if user.is_staff:
            return AnalysisJob.objects.all().prefetch_related("predictions")
        return AnalysisJob.objects.filter(user=user).prefetch_related("predictions")

    def perform_create(self, serializer):
        job = serializer.save(user=self.request.user, status="PENDING")
        try:
            task = process_analysis_job.delay(job.id)
            job.task_id = task.id
            job.save(update_fields=["task_id"])
        except Exception as e:
            job.error_message = f"Celery queue error: {e}"
            job.save(update_fields=["error_message"])

    @action(detail=False, methods=["post"], url_path="test-camera")
    def test_camera(self, request):
        """
        IP Webcam 스트림 URL 실시간 연결 테스트 API
        """
        from .serializers import validate_allowed_camera_url
        camera_url = request.data.get("camera_url", "").strip()
        if not camera_url:
            return Response({"success": False, "message": "카메라 URL을 입력해 주세요."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            validate_allowed_camera_url(camera_url)
        except Exception as e:
            return Response({"success": False, "message": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        import cv2
        try:
            cap = cv2.VideoCapture(camera_url)
            if not cap.isOpened():
                return Response({
                    "success": False,
                    "message": "카메라에 연결할 수 없습니다. URL 주소 및 스마트폰과 PC가 같은 Wi-Fi에 연결되어 있는지 확인해 주세요."
                })
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return Response({
                    "success": False,
                    "message": "카메라에 접속했으나 영상을 읽어오지 못했습니다. 앱에서 스트리밍(Start server)이 켜져 있는지 확인해 주세요."
                })
            h, w = frame.shape[:2]
            return Response({
                "success": True,
                "message": f"카메라 연결 성공! (실시간 영상 수신 중: {w}x{h})"
            })
        except Exception as e:
            return Response({
                "success": False,
                "message": f"연결 테스트 중 오류 발생: {str(e)}"
            })

    @action(detail=True, methods=["get"])

    def status(self, request, pk=None):
        job = self.get_object()
        return Response({
            "id": job.id,
            "status": job.status,
            "status_display": job.get_status_display(),
            "error_message": job.error_message,
            "is_completed": job.status in ["SUCCESS", "FAILURE"],
            "prediction_count": job.predictions.count()
        })

from django.http import StreamingHttpResponse, HttpResponseBadRequest

from ultralytics import YOLO

# YOLO-Pose 모델 1회 전역 로드로 메모리 및 지연 방지
_pose_model = None

def get_pose_model():
    global _pose_model
    if _pose_model is None:
        from pathlib import Path
        m11 = (Path(__file__).resolve().parents[2] / 'ai_service' / 'models' / 'yolo11m-pose.pt')
        _pose_model = YOLO(str(m11) if m11.exists() else 'yolov8n-pose.pt')
    return _pose_model

@login_required
def camera_stream_proxy(request):
    """
    IP Webcam 실시간 영상에 YOLO-Pose 스켈레톤(관절+뼈대)을 합성하여 초경량 고속 중계하는 뷰
    - 480px 경량 리사이즈 및 품질 60 압축으로 Wi-Fi 지연 및 끊김 완전 해소
    """
    camera_url = request.GET.get("url", "").strip()
    skeleton = request.GET.get("skeleton", "1") == "1"
    if not camera_url:
        return HttpResponseBadRequest("카메라 URL이 누락되었습니다.")

    import cv2
    import time

    def gen_frames():
        model = get_pose_model() if skeleton else None
        cap = cv2.VideoCapture(camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 최신 프레임만 즉시 획득
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                # 480px 경량화로 실시간 전송 대역폭 60% 절감
                h, w = frame.shape[:2]
                if w > 480:
                    scale = 480.0 / w
                    frame = cv2.resize(frame, (480, int(h * scale)))

                if skeleton and model is not None:
                    results = model(frame, verbose=False, conf=0.25, iou=0.45)
                    if results and len(results) > 0:
                        frame = results[0].plot(kpt_line=True, labels=False, boxes=True)

                # JPEG 품질 60 압축 (초고속 브라우저 렌더링)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if not ret:
                    continue
                header = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                yield header + buffer.tobytes() + b'\r\n'
        finally:
            cap.release()

    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
