import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from src.sequence_utils import (
    resample_sequence,
    flatten_sequence,
)
from src.feature_utils import (
    normalize_by_shoulder_width,
    selected_joint_angles,
    anatomical_body_scale,
)

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_CONFIG_PATH = BASE_DIR / "project.json"


def detect_video_rotation(video_path: Path, model: YOLO, device) -> Optional[int]:
    cap = cv2.VideoCapture(str(video_path))
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


class PosePipeline:
    def __init__(self):
        with open(PROJECT_CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        pose_model_path = BASE_DIR / self.config["pose_model"]
        raw_device = self.config.get("device", "auto")
        if raw_device == "auto":
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = raw_device

        self.model = YOLO(str(pose_model_path))
        self.last_kinematics: Dict[int, Dict] = {}
        print(f"[PosePipeline] YOLO Pose 로드 완료: {pose_model_path.name} (device={self.device})")

    def _reset_tracker(self):
        if getattr(self.model, "predictor", None) is not None:
            try:
                for tracker in getattr(self.model.predictor, "trackers", []):
                    tracker.reset()
            except Exception:
                pass

    @staticmethod
    def _clean_and_interp_keypoints(kpts_arr: np.ndarray, confs_arr: np.ndarray, min_conf: float = 0.15) -> np.ndarray:
        T = len(kpts_arr)
        out = kpts_arr.copy()
        idx = np.arange(T)
        for j in range(17):
            bad = ((np.abs(out[:, j, 0]) < 1.0) & (np.abs(out[:, j, 1]) < 1.0)) | (confs_arr[:, j] < min_conf)
            if np.any(bad) and not np.all(bad):
                good = ~bad
                out[bad, j, 0] = np.interp(idx[bad], idx[good], out[good, j, 0])
                out[bad, j, 1] = np.interp(idx[bad], idx[good], out[good, j, 1])
        return out

    @staticmethod
    def _stitch_broken_tracks(tracks: Dict[int, Dict], fps: float) -> Dict[int, Dict]:
        if len(tracks) <= 1:
            return tracks
        tids = sorted(tracks.keys(), key=lambda t: tracks[t]["fids"][0])
        merged, used = {}, set()
        max_gap = int(round(1.0 * fps))

        for tid_a in tids:
            if tid_a in used:
                continue
            cur = {k: list(v) for k, v in tracks[tid_a].items()}
            used.add(tid_a)
            changed = True
            while changed:
                changed = False
                end_f = cur["fids"][-1]
                box_a = cur["boxes"][-1]
                cx_a, cy_a = 0.5 * (box_a[0] + box_a[2]), 0.5 * (box_a[1] + box_a[3])
                diag_a = max(float(np.hypot(box_a[2] - box_a[0], box_a[3] - box_a[1])), 40.0)
                for tid_b in tids:
                    if tid_b in used:
                        continue
                    start_b = tracks[tid_b]["fids"][0]
                    if 0 < (start_b - end_f) <= max_gap:
                        box_b = tracks[tid_b]["boxes"][0]
                        cx_b, cy_b = 0.5 * (box_b[0] + box_b[2]), 0.5 * (box_b[1] + box_b[3])
                        if float(np.hypot(cx_b - cx_a, cy_b - cy_a)) <= 1.65 * diag_a:
                            for k in cur:
                                cur[k].extend(tracks[tid_b][k])
                            used.add(tid_b)
                            changed = True
                            break
            merged[tid_a] = cur
        return merged

    def extract_person_sequences(
        self, video_path: Path, min_valid_frames: int = 8, target_steps: int = 30
    ) -> Dict[int, Tuple[np.ndarray, int]]:
        rot_code = detect_video_rotation(video_path, self.model, self.device)
        self._reset_tracker()
        self.last_kinematics = {}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 1.0 or fps > 120.0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        eff_min_frames = min(min_valid_frames, max(4, total_frames // 3)) if total_frames > 0 else min_valid_frames

        raw_tracks: Dict[int, Dict[str, list]] = {}
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)

            results = self.model.track(
                frame, persist=True, verbose=False, conf=0.20, iou=0.45, device=self.device
            )

            if results and results[0].boxes is not None and results[0].keypoints is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                xyxy_arr = boxes.xyxy.cpu().numpy()
                conf_arr = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(boxes))
                kpts_xy = results[0].keypoints.xy.cpu().numpy()
                kpts_cf = (
                    results[0].keypoints.conf.cpu().numpy()
                    if results[0].keypoints.conf is not None
                    else np.ones((len(boxes), 17), dtype=np.float32)
                )

                for i in range(len(boxes)):
                    kcf = kpts_cf[i]
                    core_conf = float(np.mean(kcf[[0, 5, 6, 11, 12]]))
                    limb_conf = float(np.mean(kcf[[7, 8, 9, 10, 13, 14, 15, 16]]))
                    if core_conf < 0.32 or (core_conf < 0.45 and limb_conf < 0.20):
                        continue

                    track_id = int(boxes.id[i]) if (boxes.id is not None and len(boxes.id) > i) else 1
                    x1, y1, x2, y2 = xyxy_arr[i]
                    area = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

                    if track_id not in raw_tracks:
                        raw_tracks[track_id] = {
                            "fids": [], "kpts": [], "kconfs": [], "boxes": [], "areas": [], "bconfs": []
                        }
                    raw_tracks[track_id]["fids"].append(frame_idx)
                    raw_tracks[track_id]["kpts"].append(kpts_xy[i].astype(np.float32))
                    raw_tracks[track_id]["kconfs"].append(kcf.astype(np.float32))
                    raw_tracks[track_id]["boxes"].append(xyxy_arr[i].astype(np.float32))
                    raw_tracks[track_id]["areas"].append(area)
                    raw_tracks[track_id]["bconfs"].append(float(conf_arr[i]))

            frame_idx += 1
        cap.release()

        stitched = self._stitch_broken_tracks(raw_tracks, fps=fps)
        valid_tids = [tid for tid, d in stitched.items() if len(d["fids"]) >= eff_min_frames]
        if not valid_tids:
            return {}

        candidates = {}
        for tid in valid_tids:
            d = stitched[tid]
            kpts_raw = np.asarray(d["kpts"], dtype=np.float32)
            kconfs_raw = np.asarray(d["kconfs"], dtype=np.float32)
            fids_arr = np.asarray(d["fids"], dtype=np.int32)
            kpts_clean = self._clean_and_interp_keypoints(kpts_raw, kconfs_raw, min_conf=0.15)

            feats, hip_mid, anat, valid_idx = self._compute_track_features(kpts_clean)
            if len(feats) < max(2, eff_min_frames):
                continue

            fids_v = fids_arr[valid_idx]
            kpts_v = kpts_clean[valid_idx]
            hip_v = hip_mid[valid_idx]
            anat_v = anat[valid_idx]
            kconfs_v = kconfs_raw[valid_idx]

            ep_feats, ep_kpts, ep_hip, ep_anat, ep_fids, _ = self._extract_smart_episode(
                feats, kpts_v, hip_v, anat_v, fids_v, fps=fps
            )
            kin = self._compute_kinematics(ep_feats, ep_kpts, ep_hip, ep_anat, fps=fps)

            # 신뢰도 가중 실제 월드 좌표 관절 이동 에너지 (가려진 책상 뒤 하체 노이즈 폭주 방지!)
            ref_u = max(float(np.percentile(anat_v, 75)), 10.0)
            if len(kpts_v) >= 2:
                j_ids = [0, 5, 6, 9, 10, 15, 16]
                w_move = np.linalg.norm(np.diff(kpts_v[:, j_ids, :], axis=0), axis=2) / ref_u
                j_ok = (np.minimum(kconfs_v[:-1, j_ids], kconfs_v[1:, j_ids]) > 0.35).astype(np.float32)
                real_step = np.sum(w_move * j_ok, axis=1)
                bw = min(len(real_step), max(1, int(round(1.5 * fps))))
                real_burst = float(np.max(np.convolve(real_step, np.ones(bw), mode="valid")))
            else:
                real_burst = 0.0

            boxes_arr = np.asarray(d["boxes"], dtype=np.float32)
            frame_w = max(float(np.max(boxes_arr[:, 2])), 320.0)
            cx_med = float(np.median(0.5 * (boxes_arr[:, 0] + boxes_arr[:, 2])))

            candidates[tid] = {
                "ep_feats": ep_feats,
                "ep_fids": ep_fids,
                "total_valid": len(feats),
                "med_area": float(np.median(d["areas"])),
                "mean_bconf": float(np.mean(d["bconfs"])),
                "ank_conf": float(np.mean(kconfs_v[:, [15, 16]])),
                "burst_energy": real_burst,
                "cx_px": cx_med,
                "max_x": frame_w,
                "kin": kin,
            }

        if not candidates:
            return {}

        max_area = max(c["med_area"] for c in candidates.values())
        max_frames = max(c["total_valid"] for c in candidates.values())
        max_ank_cf = max(c["ank_conf"] for c in candidates.values())
        global_w = max(max(c["max_x"] for c in candidates.values()), 640.0)

        # 1단계: 파편 트랙, 작은 배경 사물, 좌우 끝단 잘린 행인, 책상 뒤 하체 가려진 배경 인물 선제거
        fg_candidates = {}
        for tid, c in candidates.items():
            if len(candidates) > 1:
                cx_norm = c["cx_px"] / global_w
                if c["total_valid"] < max_frames * 0.18:
                    continue
                if c["med_area"] < max_area * 0.32:
                    continue
                if (cx_norm < 0.12 or cx_norm > 0.88) and c["med_area"] < max_area * 0.75:
                    continue
                if c["ank_conf"] < 0.25 and max_ank_cf >= 0.60:
                    continue
            fg_candidates[tid] = c

        if not fg_candidates:
            fg_candidates = candidates

        # 2단계: 유효 전경 인물들 중 주 동작 수행자 대비 움직임이 미미한 정적 착석자 배제
        max_fg_burst = max(c["burst_energy"] for c in fg_candidates.values())
        person_sequences: Dict[int, Tuple[np.ndarray, int]] = {}
        for tid, c in fg_candidates.items():
            if len(fg_candidates) > 1 and max_fg_burst >= 10.0 and c["burst_energy"] < max_fg_burst * 0.35:
                continue
            resampled = resample_sequence(c["ep_feats"], c["ep_fids"], target_steps=target_steps)
            flattened = flatten_sequence(resampled)
            person_sequences[tid] = (flattened, c["total_valid"])
            self.last_kinematics[tid] = c["kin"]

        if not person_sequences and fg_candidates:
            best_tid = max(fg_candidates.keys(), key=lambda t: fg_candidates[t]["burst_energy"])
            c = fg_candidates[best_tid]
            resampled = resample_sequence(c["ep_feats"], c["ep_fids"], target_steps=target_steps)
            person_sequences[best_tid] = (flatten_sequence(resampled), c["total_valid"])
            self.last_kinematics[best_tid] = c["kin"]

        return person_sequences

    def _compute_track_features(self, kpts_seq: np.ndarray):
        feats, hip_mids, anats, valid_idx = [], [], [], []
        for idx, kpts in enumerate(kpts_seq):
            vec = self._compute_38d_feature(kpts)
            if vec is not None:
                feats.append(vec)
                hip_mids.append(0.5 * (kpts[11] + kpts[12]))
                anats.append(anatomical_body_scale(kpts))
                valid_idx.append(idx)
        if not feats:
            return np.empty((0, 38), dtype=np.float32), np.empty((0, 2)), np.empty((0,)), np.array([], dtype=np.int32)
        return (
            np.asarray(feats, dtype=np.float32),
            np.asarray(hip_mids, dtype=np.float32),
            np.asarray(anats, dtype=np.float32),
            np.asarray(valid_idx, dtype=np.int32),
        )

    def _extract_smart_episode(
        self,
        feats: np.ndarray,
        kpts: np.ndarray,
        hip_mid: np.ndarray,
        anat: np.ndarray,
        fids: np.ndarray,
        fps: float = 30.0,
    ):
        """
        동작 유형(서서 수행하는 단발성 타격/발차기/흔들기 vs 앉기/눕기/넘어지기/웅크리기 자세 전환)을
        스켈레톤 기하학으로 자동 판별하여 최적의 시퀀스 구간을 추출합니다.
        - 직립 단발/반복 동작: 피크 에너지 중심 2.1초 핵심 동작 윈도우 추출 (긴 웹캠 정지·보행 노이즈 완전 제거)
        - 착석/와위/낙상/웅크리기: 동작 시작부터 종료 자세까지 전체 에피소드 보존 (동작 반토막 방지)
        """
        T = len(feats)
        coords = feats[:, :34].reshape((T, 17, 2))
        sh_mid = (coords[:, 5, :] + coords[:, 6, :]) / 2.0
        torso_s = np.maximum(np.linalg.norm(sh_mid, axis=1), 0.40)[:, None, None]
        tc = coords / torso_s

        ref_unit = max(float(np.percentile(anat, 75)), 10.0)
        limbs = [0, 7, 8, 9, 10, 13, 14, 15, 16]
        if T >= 2:
            limb_vel = np.linalg.norm(np.diff(tc[:, limbs, :], axis=0), axis=(1, 2))
            ang_vel = np.sum(np.abs(np.diff(feats[:, 34:], axis=0)), axis=1) * 0.02
            step_e = limb_vel + ang_vel
        else:
            step_e = np.zeros(max(1, T - 1), dtype=np.float32)

        # 트랙 대표 활동성 점수: 1.5초 구간 내 최대 누적 관절 운동 에너지 (짧은 발차기·밀치기도 정확히 포착!)
        b_win = min(len(step_e), max(1, int(round(1.5 * fps))))
        burst_energy = float(np.max(np.convolve(step_e, np.ones(b_win), mode="valid"))) if len(step_e) >= b_win else float(np.sum(step_e))

        win_2s = int(np.clip(round(2.1 * fps), 45, 115))
        if T <= win_2s:
            return feats, kpts, hip_mid, anat, fids, burst_energy

        # 자세 전환(앉기·눕기·넘어지기·웅크리기) 여부 판별
        min_knee_t = np.minimum(feats[:, 36], feats[:, 37])
        ank_mid_y = 0.5 * (tc[:, 15, 1] + tc[:, 16, 1])
        sp = 0.5 * (tc[:, 5, :] + tc[:, 6, :])
        sp_tilt = np.degrees(np.arctan2(np.abs(sp[:, 0]), np.maximum(-sp[:, 1], 1e-3)))

        low_posture_frames = int(np.sum((min_knee_t < 122.0) | (ank_mid_y < 0.65) | (sp_tilt > 52.0)))
        is_posture_transition = low_posture_frames >= max(10, int(round(0.75 * fps)))

        if not is_posture_transition:
            # [직립 동작 모드] 2.1초 슬라이딩 윈도우로 카메라 시작/종료 보행 노이즈를 배제하고 핵심 동작만 정밀 추출
            win_energy = np.convolve(step_e, np.ones(win_2s - 1), mode="valid")
            if T > int(4.0 * fps):
                trim = min(int(1.0 * fps), len(win_energy) // 4)
                if trim > 0:
                    win_energy[:trim] *= 0.15
                    win_energy[-trim:] *= 0.15

            ank_span = np.maximum(tc[:, 15, 1], tc[:, 16, 1])
            med_ank = float(np.median(ank_span))
            if med_ank > 1.1:
                for s in range(len(win_energy)):
                    e = s + win_2s
                    if np.mean(tc[s:e, 0, 1] < -0.7) > 0.7 and np.mean(ank_span[s:e] < 0.65 * med_ank) > 0.4:
                        win_energy[s] *= 0.1

            best_s = int(np.argmax(win_energy))
            best_e = best_s + win_2s
            return feats[best_s:best_e], kpts[best_s:best_e], hip_mid[best_s:best_e], anat[best_s:best_e], fids[best_s:best_e], burst_energy

        # [자세 전환 모드] 시작 직립부터 착석/와위/기립까지 전체 에피소드 보존 (앞뒤 정지 구간만 제거)
        hip_vel = np.linalg.norm(np.diff(hip_mid, axis=0), axis=1) / ref_unit * 2.0
        ep_step = np.concatenate([[0.0], step_e + hip_vel])
        sm_w = max(3, int(round(0.20 * fps)))
        sm_e = np.convolve(ep_step, np.ones(sm_w) / sm_w, mode="same")
        if T > int(6.5 * fps):
            b_edge = int(0.7 * fps)
            sm_e[:b_edge] *= 0.25
            sm_e[-b_edge:] *= 0.25

        peak_e = float(np.max(sm_e))
        if peak_e >= 0.04:
            thr = max(0.14 * peak_e, float(np.percentile(sm_e, 20)) * 1.4)
            act = np.where(sm_e >= thr)[0]
            if len(act) >= 3:
                pre, post = int(round(0.35 * fps)), int(round(0.65 * fps))
                s_idx = max(0, int(act[0]) - pre)
                e_idx = min(T, int(act[-1]) + post + 1)
                if e_idx - s_idx >= win_2s:
                    return feats[s_idx:e_idx], kpts[s_idx:e_idx], hip_mid[s_idx:e_idx], anat[s_idx:e_idx], fids[s_idx:e_idx], burst_energy

        return feats, kpts, hip_mid, anat, fids, burst_energy

    @staticmethod
    def _compute_kinematics(feats: np.ndarray, kpts: np.ndarray, hip_mid: np.ndarray, anat: np.ndarray, fps: float) -> Dict:
        L = len(feats)
        ref_unit = max(float(np.percentile(anat, 75)), 10.0)
        init_n = max(2, L // 7)
        hip_y = (hip_mid[:, 1] - float(np.median(hip_mid[:init_n, 1]))) / ref_unit
        hip_x = (hip_mid[:, 0] - float(np.median(hip_mid[:init_n, 0]))) / ref_unit
        nose_y = (kpts[:, 0, 1] - float(np.median(kpts[:init_n, 0, 1]))) / ref_unit
        step_k = max(2, min(max(L - 1, 2), int(round(0.25 * fps))))
        dt_k = max(step_k / max(fps, 1.0), 0.05)
        nose_vy_peak = float(np.max(nose_y[step_k:] - nose_y[:-step_k]) / dt_k) if L > step_k else 0.0
        return {
            "dur_sec": L / max(fps, 1.0),
            "hip_drop_max": float(np.percentile(hip_y, 95)),
            "hip_x_span": float(np.percentile(hip_x, 95) - np.percentile(hip_x, 5)),
            "nose_vy_peak": nose_vy_peak,
        }

    def _compute_38d_feature(self, kpts: np.ndarray) -> Optional[np.ndarray]:
        try:
            norm_kpts, _, _ = normalize_by_shoulder_width(kpts)
            angles = selected_joint_angles(kpts)
            vec = []
            for pt in norm_kpts:
                vec.extend([float(pt[0]), float(pt[1])])
            vec.extend([
                float(angles["left_elbow_angle"]),
                float(angles["right_elbow_angle"]),
                float(angles["left_knee_angle"]),
                float(angles["right_knee_angle"]),
            ])
            arr = np.asarray(vec, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=160.0)
            if len(arr) == 38 and np.all(np.isfinite(arr)):
                return arr
        except Exception:
            pass
        return None


pose_pipeline = PosePipeline()
