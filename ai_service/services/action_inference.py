from typing import Dict, List, Tuple
import numpy as np
from scipy.signal import find_peaks

from .model_loader import model_loader
from .pose_pipeline import pose_pipeline
from src.feature_utils import extract_kinematic_summary_86d


class ActionInference:
    def infer_actions(
        self, person_sequences: Dict[int, Tuple[np.ndarray, int]]
    ) -> List[Dict]:
        if not model_loader.is_loaded:
            model_loader.load_model()

        model = model_loader.model
        if model is None:
            raise RuntimeError("Random Forest 모델이 로드되지 않았습니다.")

        predictions = []
        n_expected = int(getattr(model, "n_features_in_", 1140))

        for track_id, (seq_1140, valid_frames) in person_sequences.items():
            if n_expected == 1226:
                kin_86 = extract_kinematic_summary_86d(seq_1140)
                feat_vec = np.concatenate([seq_1140, kin_86])
            else:
                feat_vec = seq_1140

            X = np.expand_dims(feat_vec, axis=0)
            proba = model.predict_proba(X)[0].copy()
            classes_list = model.classes_.tolist()
            kin = pose_pipeline.last_kinematics.get(track_id, {})

            frames = seq_1140.reshape((30, 38))
            coords = frames[:, :34].reshape((30, 17, 2))
            sh_mid = (coords[:, 5, :] + coords[:, 6, :]) / 2.0
            torso_s = np.maximum(np.linalg.norm(sh_mid, axis=1), 0.40)[:, None, None]
            tc = coords / torso_s

            l_elb, r_elb = frames[:, 34], frames[:, 35]
            l_knee, r_knee = frames[:, 36], frames[:, 37]
            avg_elb = (l_elb + r_elb) / 2.0
            min_knee = float(np.min(np.minimum(l_knee, r_knee)))
            p10_knee = float(np.percentile(np.minimum(l_knee, r_knee), 10))
            support_knee = float(np.median(np.maximum(l_knee, r_knee)))
            max_elbow = float(np.max(np.maximum(l_elb, r_elb)))
            min_avg_elbow = float(np.min(avg_elb))
            mean_elbow = float(np.mean(avg_elb))

            mean_nose_y = float(np.mean(tc[:, 0, 1]))
            max_nose_y = float(np.max(tc[:, 0, 1]))
            l_ank_y, r_ank_y = tc[:, 15, 1], tc[:, 16, 1]
            l_ank_x, r_ank_x = tc[:, 15, 0], tc[:, 16, 0]
            ank_mid_y = (l_ank_y + r_ank_y) / 2.0
            min_ank_mid_y = float(np.min(ank_mid_y))
            standing_ank_y = float(np.median(np.maximum(l_ank_y, r_ank_y)))
            min_single_ank_y = float(np.min(np.minimum(l_ank_y, r_ank_y)))
            ank_asym_y = float(np.max(np.abs(l_ank_y - r_ank_y)))
            ank_sep_2d = float(np.max(np.hypot(l_ank_x - r_ank_x, l_ank_y - r_ank_y)))

            sh_y = (tc[:, 5, 1] + tc[:, 6, :])[:, 1] if False else 0.5 * (tc[:, 5, 1] + tc[:, 6, 1])
            l_wy, r_wy = tc[:, 9, 1], tc[:, 10, 1]
            l_wx, r_wx = tc[:, 9, 0], tc[:, 10, 0]
            w_mid_y = 0.5 * (l_wy + r_wy)
            w_mid_x = 0.5 * (l_wx + r_wx)

            l_y_tr, r_y_tr = float(np.ptp(l_wy)), float(np.ptp(r_wy))
            wrist_wy_tr = max(l_y_tr, r_y_tr)
            wrist_wx_tr = max(float(np.ptp(l_wx)), float(np.ptp(r_wx)))
            two_hand_y = min(l_y_tr, r_y_tr) / max(1e-4, wrist_wy_tr)

            mid_wy_min = float(np.min(w_mid_y))
            mid_wy_span = float(np.ptp(w_mid_y))
            mid_wx_span = float(np.ptp(w_mid_x))
            min_w_rel_sh = float(np.min(np.minimum(l_wy, r_wy) - sh_y))
            max_wrist_y = float(np.max(np.maximum(l_wy, r_wy)))

            sp = 0.5 * (tc[:, 5, :] + tc[:, 6, :])
            sp_tilt = np.degrees(np.arctan2(np.abs(sp[:, 0]), np.maximum(-sp[:, 1], 1e-3)))
            max_tilt = float(np.percentile(sp_tilt, 90))
            hip_drop = float(kin.get("hip_drop_max", max(0.0, 1.30 - min_ank_mid_y)))
            hip_x_span = float(kin.get("hip_x_span", 0.30))

            def mul(names, factor):
                for n in (names if isinstance(names, list) else [names]):
                    if n in classes_list:
                        proba[classes_list.index(n)] *= factor

            is_upright_standing = (
                mean_nose_y < -0.68
                and standing_ank_y > 0.92
                and min_ank_mid_y > 0.68
                and p10_knee > 130.0
                and max_tilt < 48.0
            )

            # A. 명확한 앞차기 시그니처 (한쪽 지지발 직립 + 반대쪽 발 비대칭 상승)
            is_clear_kick = (
                mean_nose_y < -0.68
                and standing_ank_y > 1.0
                and support_knee > 138.0
                and min_knee < 145.0
                and ank_asym_y >= 0.42
                and min_single_ank_y <= 0.85
                and max_wrist_y < 1.02
            )
            if is_clear_kick:
                mul("front_kick", 10.0)
                mul(["throw_one_hand", "sit_on_object", "sit_floor", "crouch_take_item",
                     "shotgun", "kiosk_tap", "push", "grab_shake", "swing", "body_slam"], 0.08)

            # B. 직립 상체 동작 물리 검증 (샷건 vs 잡고 흔들기 vs 밀치기 vs 키오스크 vs 물건던지기)
            elif is_upright_standing and float(np.max(proba)) < 0.72:
                mul(["lie_forward", "lie_backward", "fall_backward", "fall_forward", "fall_sideways", "sit_floor", "crouch_take_item"], 0.05)
                if min_knee > 152.0:
                    mul("sit_on_object", 0.08)
                if min_knee > 145.0 and min_single_ank_y > 0.88 and ank_sep_2d < 0.90:
                    mul("front_kick", 0.08)

                peaks_e, _ = find_peaks(avg_elb, prominence=14.0, height=102.0)
                valleys_e, _ = find_peaks(-avg_elb, prominence=14.0)
                steps_ext = int(np.sum(avg_elb > 90.0))

                # 1) 양손을 머리 높이 위로 치켜들어 수직으로 강하게 내려치는 동작 -> shotgun
                if mid_wy_min <= -1.35 and mid_wy_span >= 1.05 and two_hand_y >= 0.60 and mid_wy_span > mid_wx_span * 1.4:
                    mul("shotgun", 6.0)
                    mul(["kiosk_tap", "push", "grab_shake", "swing"], 0.20)
                # 2) 한손 머리 위 동작 -> throw_one_hand vs kiosk_tap
                elif min_w_rel_sh < -0.30 and wrist_wy_tr >= 0.65 and two_hand_y < 0.45:
                    if wrist_wx_tr >= 1.00:
                        mul("throw_one_hand", 3.2)
                        mul("kiosk_tap", 0.40)
                    else:
                        mul("kiosk_tap", 4.0)
                        mul("shotgun", 0.25)
                # 3) 전방/측면 양손 뻗기 및 흔들기 -> grab_shake vs push
                elif max_elbow >= 145.0 and two_hand_y >= 0.50:
                    is_shake = (
                        mid_wx_span < 0.55
                        and (
                            len(peaks_e) >= 2
                            or (len(valleys_e) >= 3 and min_avg_elbow < 52.0 and mean_elbow >= 65.0)
                            or (len(valleys_e) >= 2 and min_avg_elbow < 45.0 and mean_elbow >= 65.0 and steps_ext >= 8)
                        )
                    )
                    if is_shake:
                        mul("grab_shake", 4.5)
                        mul(["push", "shotgun", "swing", "throw_one_hand"], 0.20)
                    elif mid_wx_span < 0.55 or (min_avg_elbow >= 70.0 and len(peaks_e) <= 1):
                        # 정면 단발 밀치기 또는 90도 측면 전방 밀치기(min_avg_elbow >= 70, 단일 피크)
                        mul("push", 4.0)
                        mul(["grab_shake", "shotgun", "kiosk_tap"], 0.25)

            # C. 상단 CCTV 각도 수직 전방 눕기 (발목이 골반 위로 역전 min_ank_mid_y < 0.10 & 코가 골반 아래로 하강 max_nose_y > 0.80)
            elif min_ank_mid_y < 0.10 and max_nose_y > 0.80:
                mul(["crouch_take_item", "sit_on_object", "vomit", "fall_sideways", "lie_backward"], 0.12)
                mul("lie_forward", 4.0)

            total_p = float(np.sum(proba))
            if total_p > 0:
                proba /= total_p
            top1_idx = int(np.argmax(proba))
            predictions.append({
                "track_id": int(track_id),
                "action_label": classes_list[top1_idx],
                "confidence": round(float(proba[top1_idx]), 4),
                "valid_frames": int(valid_frames),
            })

        return predictions


action_inference = ActionInference()
