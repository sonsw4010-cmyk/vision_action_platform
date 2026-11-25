from urllib.parse import urlparse
import ipaddress
from rest_framework import serializers
from .models import AnalysisJob, PersonPrediction

def validate_allowed_camera_url(url: str):
    """
    체크리스트 7번 준수:
    수업에서 허용한 스마트폰 IP Webcam(사설 IP 대역: 192.168.x.x, 10.x.x.x, 172.16~31.x.x, localhost)만
    연결을 허용하고, 외부 임의 URL(SSRF 공격 등) 호출을 차단합니다.
    """
    if not (url.startswith("http://") or url.startswith("https://") or url.startswith("rtsp://")):
        raise serializers.ValidationError("유효한 스트림 URL 프로토콜(http://, https://, rtsp://)을 사용해야 합니다.")
    
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            raise serializers.ValidationError("카메라 호스트 주소를 파싱할 수 없습니다.")
        
        if hostname in ["localhost", "127.0.0.1"]:
            return True
        
        ip = ipaddress.ip_address(hostname)
        if not (ip.is_private or ip.is_loopback):
            raise serializers.ValidationError(
                f"외부 인터넷 주소({hostname})는 보안상 허용되지 않습니다. "
                "수업에서 안내된 스마트폰 로컬 Wi-Fi 사설 IP(예: 192.168.x.x)를 사용하세요."
            )
    except ValueError:
        raise serializers.ValidationError(
            f"허용되지 않은 카메라 호스트입니다: {hostname}. "
            "스마트폰 IP Webcam의 로컬 IP(예: 192.168.0.x)를 입력해 주세요."
        )
    return True

class PersonPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PersonPrediction
        fields = [
            "id", "track_id", "action_label", "confidence",
            "valid_frames", "created_at"
        ]
        read_only_fields = fields

class AnalysisJobSerializer(serializers.ModelSerializer):
    predictions = PersonPredictionSerializer(many=True, read_only=True)
    username = serializers.ReadOnlyField(source="user.username")

    class Meta:
        model = AnalysisJob
        fields = [
            "id", "user", "username", "title", "input_type",
            "video_file", "camera_url", "status", "expected_action",
            "task_id", "error_message", "predictions", "created_at",
            "completed_at"
        ]
        read_only_fields = [
            "id", "user", "status", "task_id", "error_message",
            "predictions", "created_at", "completed_at"
        ]

    def validate(self, attrs):
        input_type = attrs.get("input_type", getattr(self.instance, "input_type", "upload"))
        
        if not self.instance:
            if input_type == "upload":
                if not attrs.get("video_file"):
                    raise serializers.ValidationError({"video_file": "영상 파일 업로드 방식에서는 video_file이 필수입니다."})
            elif input_type == "camera":
                camera_url = attrs.get("camera_url", "").strip()
                if not camera_url:
                    raise serializers.ValidationError({"camera_url": "IP Webcam 방식에서는 camera_url이 필수입니다."})
                validate_allowed_camera_url(camera_url)
        return attrs
