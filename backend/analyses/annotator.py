import os
import subprocess
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

_annotator_pose_model = None
_annotator_device = None


def get_annotator_model():
    global _annotator_pose_model, _annotator_device
    if _annotator_pose_model is None:
        _annotator_device = 0 if torch.cuda.is_available() else "cpu"
        m11_path = (Path(__file__).resolve().parents[2] / "ai_service" / "models" / "yolo11m-pose.pt")
        model_file = str(m11_path) if m11_path.exists() else "yolov8n-pose.pt"
        _annotator_pose_model = YOLO(model_file)
    return _annotator_pose_model, _annotator_device


def _detect_rotation_for_annotator(input_path: str, model: YOLO, device):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idxs = sorted(list(set([max(0, int(nf * p)) for p in [0.02, 0.12, 0.28, 0.50, 0.72]])))
    want = set(sample_idxs)
    frames, idx = [], 0
    while cap.isOpened() and len(frames) < len(sample_idxs):
        ret, fr = cap.read()
        if not ret:
            break
        if idx in want:
            frames.append(fr)
        idx += 1
    cap.release()
    if not frames:
        return None

    def eval_rot(rot_code):
        up_scores, raw_confs = [], []
        horiz_cnt, det_cnt = 0, 0
        for fr in frames:
            im = cv2.rotate(fr, rot_code) if rot_code is not None else fr
            res = model(im, verbose=False, conf=0.20, device=device)
            if res and res[0].boxes is not None and len(res[0].boxes) > 0 and res[0].keypoints is not None:
                confs_b = res[0].boxes.conf.cpu().numpy()
                bi = int(np.argmax(confs_b))
                bconf = float(confs_b[bi])
                kxy = res[0].keypoints.xy.cpu().numpy()[bi]
                kcf = res[0].keypoints.conf.cpu().numpy()[bi] if res[0].keypoints.conf is not None else np.ones(17)
                core_cf = float(np.mean(kcf[[0, 5, 6, 11, 12]]))
                if core_cf < 0.25:
                    up_scores.append(0.0); raw_confs.append(0.0); continue
                sh_mid = (kxy[5] + kxy[6]) / 2.0
                hp_mid = (kxy[11] + kxy[12]) / 2.0
                dx = abs(float(sh_mid[0] - hp_mid[0]))
                dy = float(hp_mid[1] - sh_mid[1])
                det_cnt += 1
                raw_confs.append(bconf * core_cf)
                if dx > 1.25 * abs(dy):
                    horiz_cnt += 1
                up_factor = 1.0 if dy > 0.45 * dx else (0.18 if dy > 0 else 0.04)
                up_scores.append(bconf * core_cf * up_factor)
            else:
                up_scores.append(0.0); raw_confs.append(0.0)
        return float(np.mean(up_scores)), float(np.mean(raw_confs)), horiz_cnt, det_cnt

    s0, c0, h0, d0 = eval_rot(None)
    if s0 >= 0.40 and h0 < max(2, len(frames) - 1):
        return None
    best_rot, best_s = None, s0
    for r_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
        sr, cr, _, dr = eval_rot(r_code)
        if sr > best_s + 0.18 and sr > max(s0 * 1.65, 0.35) and cr >= c0 * 0.90 and dr >= max(1, d0):
            best_s = sr
            best_rot = r_code
    return best_rot


SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


def create_annotated_video(input_path: str, output_path: str, predictions_map: dict = None) -> str:
    """
    AI 추론 파이프라인과 동일한 YOLO11m-Pose + ByteTrack + 자동 회전 보정 + 의자 오검출 필터를 사용하여
    모든 프레임에 정확한 관절 스켈레톤과 동작 예측 라벨을 렌더링합니다.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {input_path}")

    model, device = get_annotator_model()
    rot_code = _detect_rotation_for_annotator(input_path, model, device)

    if getattr(model, "predictor", None) is not None:
        try:
            for tracker in getattr(model.predictor, "trackers", []):
                tracker.reset()
        except Exception:
            pass

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1.0 or fps > 120.0:
        fps = 25.0

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError(f"영상을 읽을 수 없습니다: {input_path}")
    if rot_code is not None:
        first_frame = cv2.rotate(first_frame, rot_code)
    h, w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    single_fallback_label = list(predictions_map.values())[0] if (predictions_map and len(predictions_map) == 1) else None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)

            results = model.track(frame, persist=True, verbose=False, conf=0.20, iou=0.45, device=device)
            if results and results[0].boxes is not None and results[0].keypoints is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                xyxy_arr = boxes.xyxy.cpu().numpy()
                kpts_xy = results[0].keypoints.xy.cpu().numpy()
                kpts_cf = (
                    results[0].keypoints.conf.cpu().numpy()
                    if results[0].keypoints.conf is not None
                    else np.ones((len(boxes), 17), dtype=np.float32)
                )

                valid_indices = []
                for i in range(len(boxes)):
                    core_cf = float(np.mean(kpts_cf[i, [0, 5, 6, 11, 12]]))
                    limb_cf = float(np.mean(kpts_cf[i, [7, 8, 9, 10, 13, 14, 15, 16]]))
                    if core_cf >= 0.32 and not (core_cf < 0.45 and limb_cf < 0.20):
                        valid_indices.append(i)

                # 가장 큰 주 피사체 판별 (단일 예측 라벨 폴백 매칭용)
                primary_i = (
                    max(valid_indices, key=lambda idx: (xyxy_arr[idx][2] - xyxy_arr[idx][0]) * (xyxy_arr[idx][3] - xyxy_arr[idx][1]))
                    if valid_indices else -1
                )

                for i in valid_indices:
                    track_id = int(boxes.id[i]) if (boxes.id is not None and len(boxes.id) > i) else 1
                    x1, y1, x2, y2 = xyxy_arr[i].astype(int)
                    kxy = kpts_xy[i]
                    kcf = kpts_cf[i]

                    # 바운딩 박스 및 관절 뼈대 선명하게 렌더링
                    is_labeled = bool(predictions_map and (track_id in predictions_map or (single_fallback_label and i == primary_i)))
                    box_color = (0, 230, 118) if is_labeled else (180, 180, 180)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    for u, v in SKELETON_EDGES:
                        if kcf[u] >= 0.20 and kcf[v] >= 0.20:
                            pu, pv = (int(kxy[u, 0]), int(kxy[u, 1])), (int(kxy[v, 0]), int(kxy[v, 1]))
                            if pu != (0, 0) and pv != (0, 0):
                                cv2.line(frame, pu, pv, (255, 190, 0), 2, cv2.LINE_AA)

                    for j in range(17):
                        if kcf[j] >= 0.20:
                            pj = (int(kxy[j, 0]), int(kxy[j, 1]))
                            if pj != (0, 0):
                                cv2.circle(frame, pj, 4, (0, 255, 255), -1, cv2.LINE_AA)

                    label_text = None
                    if predictions_map:
                        label_text = predictions_map.get(track_id)
                        if label_text is None and single_fallback_label and i == primary_i:
                            label_text = single_fallback_label

                    if label_text:
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
                        ty = max(th + 10, y1 - 8)
                        cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 10, ty + 4), (0, 140, 70), -1)
                        cv2.putText(
                            frame, label_text, (x1 + 5, ty - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA
                        )

            out.write(frame)
    finally:
        cap.release()
        out.release()

    h264_path = str(output_path) + ".h264.mp4"
    try:
        ret = subprocess.run([
            "ffmpeg", "-y", "-i", str(output_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            h264_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if ret.returncode == 0 and os.path.exists(h264_path):
            os.replace(h264_path, str(output_path))
    except Exception:
        pass

    return output_path
