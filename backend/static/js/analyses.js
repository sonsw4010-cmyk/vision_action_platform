// Vision Action AI Frontend Script
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadJobForm');
    const cameraForm = document.getElementById('cameraJobForm');
    const liveProgressCard = document.getElementById('liveProgressCard');
    const progressStatusBadge = document.getElementById('progressStatusBadge');
    const progressMessage = document.getElementById('progressMessage');
    const progressViewBtn = document.getElementById('progressViewBtn');

    let activePollingInterval = null;

    function startStatusPolling(jobId) {
        liveProgressCard.classList.remove('d-none');
        liveProgressCard.scrollIntoView({ behavior: 'smooth' });

        if (activePollingInterval) clearInterval(activePollingInterval);

        activePollingInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/v1/analyses/${jobId}/status/`);
                if (!res.ok) return;
                const data = await res.json();

                progressStatusBadge.textContent = data.status_display;
                progressStatusBadge.className = `badge badge-status badge-status-${data.status}`;

                if (data.status === 'PROCESSING') {
                    progressMessage.innerHTML = '<div class="spinner-border spinner-border-sm text-primary me-2"></div>YOLO Pose 추출 및 17개 동작 추론 진행 중...';
                } else if (data.status === 'SUCCESS') {
                    clearInterval(activePollingInterval);
                    progressMessage.innerHTML = `<span class="text-success"><i class="bi bi-check-circle-fill me-1"></i>분석 완료! (${data.prediction_count}명 검출됨)</span>`;
                    progressViewBtn.href = `/analyses/${jobId}/`;
                    progressViewBtn.classList.remove('d-none');
                } else if (data.status === 'FAILURE') {
                    clearInterval(activePollingInterval);
                    progressMessage.innerHTML = `<span class="text-danger"><i class="bi bi-exclamation-triangle-fill me-1"></i>실패: ${data.error_message || '알 수 없는 오류'}</span>`;
                    progressViewBtn.href = `/analyses/${jobId}/`;
                    progressViewBtn.classList.remove('d-none');
                }
            } catch (err) {
                console.error("Polling error:", err);
            }
        }, 2000);
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const submitBtn = uploadForm.querySelector('button[type="submit"]');
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>업로드 중...';

            const formData = new FormData(uploadForm);
            formData.append('input_type', 'upload');

            try {
                const res = await fetch('/api/v1/analyses/', {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': getCookie('csrftoken')
                    },
                    body: formData
                });
                const result = await res.json();
                if (res.ok) {
                    startStatusPolling(result.id);
                    uploadForm.reset();
                } else {
                    alert('등록 오류: ' + JSON.stringify(result));
                }
            } catch (err) {
                alert('네트워크 오류가 발생했습니다: ' + err.message);
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="bi bi-upload me-1"></i>영상 분석 시작';
            }
        });
    }

    // IP Webcam 실시간 연결 테스트 핸들러
    const btnTestCamera = document.getElementById('btnTestCamera');
    const cameraUrlInput = document.getElementById('cameraUrl');
    const cameraTestResult = document.getElementById('cameraTestResult');

    if (btnTestCamera) {
        btnTestCamera.addEventListener('click', async () => {
            const url = cameraUrlInput.value.trim();
            if (!url) {
                alert('카메라 스트림 URL을 먼저 입력해 주세요.');
                cameraUrlInput.focus();
                return;
            }

            btnTestCamera.disabled = true;
            btnTestCamera.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>확인 중...';
            cameraTestResult.className = 'alert alert-info small mt-2';
            cameraTestResult.classList.remove('d-none');
            cameraTestResult.innerHTML = '<div class="spinner-border spinner-border-sm me-2"></div>카메라 스트림 신호를 확인하는 중입니다 (약 2~4초)...';

            try {
                const res = await fetch('/api/v1/analyses/test-camera/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCookie('csrftoken')
                    },
                    body: JSON.stringify({ camera_url: url })
                });
                const data = await res.json();
                if (data.success) {
                    cameraTestResult.className = 'alert alert-success small mt-2';
                    cameraTestResult.innerHTML = `<i class="bi bi-check-circle-fill me-1 text-success"></i><strong>${data.message}</strong>`;
                    
                    // 실시간 미리보기 화면 활성화 (Skeleton Overlay 지원)
                    const previewContainer = document.getElementById('cameraPreviewContainer');
                    const liveImg = document.getElementById('cameraLivePreview');
                    const toggleSkeleton = document.getElementById('toggleSkeleton');

                    function updateStreamUrl() {
                        const isSkeleton = toggleSkeleton ? toggleSkeleton.checked : true;
                        if (isSkeleton) {
                            // 스켈레톤 합성 모드 (서버 초경량 프록시)
                            liveImg.src = `/analyses/camera-stream/?url=${encodeURIComponent(url)}&skeleton=1`;
                        } else {
                            // 원본 모드: 스마트폰 1:1 직결로 지연 0ms, 초부드러운 60 FPS 재생
                            liveImg.src = url;
                        }
                    }

                    if (previewContainer && liveImg) {
                        previewContainer.classList.remove('d-none');
                        updateStreamUrl();

                        if (toggleSkeleton) {
                            toggleSkeleton.onchange = () => {
                                updateStreamUrl();
                            };
                        }
                    }
                } else {
                    cameraTestResult.className = 'alert alert-danger small mt-2';
                    cameraTestResult.innerHTML = `
                        <div class="fw-bold mb-1"><i class="bi bi-exclamation-triangle-fill me-1"></i>${data.message}</div>
                        <ul class="mb-0 ps-3">
                            <li>스마트폰 앱 맨 아래 <strong>Start server</strong>를 켜셨나요?</li>
                            <li>주소 끝에 <strong>/video</strong> 가 붙어 있나요? (예: <code>http://192.168.0.15:8080/video</code>)</li>
                            <li>스마트폰과 PC가 <strong>같은 Wi-Fi</strong>에 연결되어 있나요?</li>
                        </ul>
                    `;
                }
            } catch (err) {
                cameraTestResult.className = 'alert alert-danger small mt-2';
                cameraTestResult.innerHTML = `<i class="bi bi-x-circle me-1"></i>연결 요청 실패: ${err.message}`;
            } finally {
                btnTestCamera.disabled = false;
                btnTestCamera.innerHTML = '<i class="bi bi-broadcast me-1"></i>연결 테스트';
            }
        });
    }

    if (cameraForm) {
        cameraForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const submitBtn = cameraForm.querySelector('button[type="submit"]');
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>카메라 수집 시작...';

            const formData = new FormData(cameraForm);
            formData.append('input_type', 'camera');

            try {
                const res = await fetch('/api/v1/analyses/', {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': getCookie('csrftoken')
                    },
                    body: formData
                });
                const result = await res.json();
                if (res.ok) {
                    startStatusPolling(result.id);
                    cameraForm.reset();
                } else {
                    alert('등록 오류: ' + JSON.stringify(result));
                }
            } catch (err) {
                alert('네트워크 오류가 발생했습니다: ' + err.message);
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="bi bi-camera-video me-1"></i>스트림 수집 및 분석 시작';
            }
        });
    }

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});
