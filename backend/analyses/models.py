from django.db import models
from django.conf import settings

# 공식 6개 동작 한글 매핑 딕셔너리
ACTION_KOREAN_DICT = {
    "kiosk_tap": "키오스크 내려치기 (A001)",
    "throw_one_hand": "한손 물건 던지기 (A002)",
    "front_kick": "앞차기 (A003)",
    "grab_shake": "잡고 흔들기 (A004)",
    "push": "밀치기 (A005)",
    "swing": "휘두르기 (A006)",
    "sit_floor": "바닥에 앉기 (A007)",
    "sit_on_object": "위에 앉기 (A008)",
    "lie_forward": "앞으로 눕기 (A009)",
    "lie_backward": "뒤로 눕기 (A010)",
    "body_slam": "몸통 박치기 (A011)",
    "crouch_take_item": "쪼그려 물건 빼내기 (A012)",
    "shotgun": "샷건 (A013)",
    "vomit": "구토 (A014)",
    "fall_sideways": "옆으로 쓰러지기 (A015)",
    "fall_backward": "뒤로 넘어지기 (A016)",
    "fall_forward": "앞으로 넘어지기 (A017)",
}

class AnalysisJob(models.Model):
    INPUT_TYPE_CHOICES = (
        ("upload", "영상 업로드"),
        ("camera", "IP Webcam"),
    )
    STATUS_CHOICES = (
        ("PENDING", "대기 중"),
        ("PROCESSING", "분석 진행 중"),
        ("SUCCESS", "분석 완료"),
        ("FAILURE", "분석 실패"),
    )

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="analysis_jobs",
        verbose_name="요청 사용자",
    )
    title = models.CharField(max_length=200, default="동작 분석 작업", verbose_name="작업 제목")
    input_type = models.CharField(
        max_length=20, choices=INPUT_TYPE_CHOICES, default="upload", verbose_name="입력 방식"
    )
    video_file = models.FileField(
        upload_to="videos/%Y/%m/%d/", null=True, blank=True, verbose_name="영상 파일"
    )
    annotated_video = models.FileField(
        upload_to="annotated_videos/%Y/%m/%d/", null=True, blank=True, verbose_name="스켈레톤 분석 영상"
    )
    camera_url = models.CharField(
        max_length=500, null=True, blank=True, verbose_name="카메라 스트림 URL"
    )
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="PENDING", verbose_name="작업 상태"
    )
    expected_action = models.CharField(
        max_length=100, null=True, blank=True, verbose_name="사용자 선택 예상 동작"
    )
    task_id = models.CharField(
        max_length=255, null=True, blank=True, verbose_name="Celery Task ID"
    )
    error_message = models.TextField(null=True, blank=True, verbose_name="오류 사유")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성 시각")
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name="완료 시각")

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "분석 작업"
        verbose_name_plural = "분석 작업 목록"

    def __str__(self):
        return f"[{self.get_status_display()}] {self.title} (ID: {self.id})"


class PersonPrediction(models.Model):
    job = models.ForeignKey(
        AnalysisJob,
        on_delete=models.CASCADE,
        related_name="predictions",
        verbose_name="소속 작업",
    )
    track_id = models.IntegerField(default=1, verbose_name="사람 Track ID")
    action_label = models.CharField(max_length=100, verbose_name="예측 동작")
    confidence = models.FloatField(default=0.0, verbose_name="신뢰도 (Confidence)")
    valid_frames = models.IntegerField(default=0, verbose_name="유효 관절 프레임 수")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="기록 시각")

    class Meta:
        ordering = ["track_id"]
        verbose_name = "사람별 예측 결과"
        verbose_name_plural = "사람별 예측 결과 목록"

    @property
    def action_korean(self):
        return ACTION_KOREAN_DICT.get(self.action_label, self.action_label)

    @property
    def confidence_pct(self):
        return self.confidence * 100.0 if self.confidence <= 1.0 else self.confidence

    def __str__(self):
        return f"Job #{self.job_id} | Track {self.track_id}: {self.action_korean} ({self.confidence * 100:.1f}%)"
