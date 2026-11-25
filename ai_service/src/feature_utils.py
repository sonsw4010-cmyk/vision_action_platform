from __future__ import annotations

import math
import numpy as np
from scipy.signal import find_peaks

NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

LEFT_RIGHT_PAIRS = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
]


def calculate_angle(point_a, point_b, point_c) -> float:
    a = np.asarray(point_a, dtype=np.float32)
    b = np.asarray(point_b, dtype=np.float32)
    c = np.asarray(point_c, dtype=np.float32)

    vector_ba = a - b
    vector_bc = c - b
    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 160.0

    cosine = np.dot(vector_ba, vector_bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(math.degrees(math.acos(float(cosine))))


def midpoint(point_a, point_b):
    a = np.asarray(point_a, dtype=np.float32)
    b = np.asarray(point_b, dtype=np.float32)
    return (a + b) / 2.0


def body_center(keypoints):
    return midpoint(keypoints[LEFT_HIP], keypoints[RIGHT_HIP])


def shoulder_width(keypoints) -> float:
    left = np.asarray(keypoints[LEFT_SHOULDER], dtype=np.float32)
    right = np.asarray(keypoints[RIGHT_SHOULDER], dtype=np.float32)
    return float(np.linalg.norm(left - right))


def anatomical_body_scale(keypoints) -> float:
    """
    카메라 각도(정면, 대각선, 90도 측면, 눕기/쓰러지기 단축 투영)에 강건한 신체 기준 길이.
    정면에서는 기존 어깨 폭(shoulder_width)을 그대로 사용하고,
    측면/누운 자세에서 어깨 폭이 0에 수렴할 때 몸통·대퇴·상완 길이 기반 하한선으로 좌표 폭주를 방지합니다.
    """
    kpts = np.asarray(keypoints, dtype=np.float32)
    sw = shoulder_width(kpts)
    sh_mid = midpoint(kpts[LEFT_SHOULDER], kpts[RIGHT_SHOULDER])
    hp_mid = midpoint(kpts[LEFT_HIP], kpts[RIGHT_HIP])
    torso = float(np.linalg.norm(sh_mid - hp_mid))
    l_thigh = float(np.linalg.norm(kpts[LEFT_HIP] - kpts[LEFT_KNEE]))
    r_thigh = float(np.linalg.norm(kpts[RIGHT_HIP] - kpts[RIGHT_KNEE]))
    l_arm = float(np.linalg.norm(kpts[LEFT_SHOULDER] - kpts[LEFT_ELBOW]))
    r_arm = float(np.linalg.norm(kpts[RIGHT_SHOULDER] - kpts[RIGHT_ELBOW]))

    anat = max(torso, 1.15 * max(l_thigh, r_thigh), 1.45 * max(l_arm, r_arm), 10.0)
    return float(max(sw, 0.52 * anat))


def center_keypoints(keypoints):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    center = body_center(keypoints)
    centered = keypoints - center
    return centered, center


def normalize_by_shoulder_width(keypoints):
    """
    골반 중심 원점 이동 후, 측면/누운 자세 특이점이 보정된 어깨 폭 기준으로 크기를 정규화합니다.
    """
    centered, center = center_keypoints(keypoints)
    scale = anatomical_body_scale(keypoints)
    if scale < 1e-8:
        raise ValueError("신체 기준 길이를 계산할 수 없습니다.")
    normalized = centered / scale
    return normalized, center, scale


def flip_keypoints_horizontal(normalized_keypoints):
    flipped = np.asarray(normalized_keypoints, dtype=np.float32).copy()
    flipped[:, 0] *= -1.0
    for left_id, right_id in LEFT_RIGHT_PAIRS:
        temp = flipped[left_id].copy()
        flipped[left_id] = flipped[right_id]
        flipped[right_id] = temp
    return flipped


def selected_joint_angles(keypoints):
    return {
        "left_elbow_angle": calculate_angle(
            keypoints[LEFT_SHOULDER], keypoints[LEFT_ELBOW], keypoints[LEFT_WRIST]
        ),
        "right_elbow_angle": calculate_angle(
            keypoints[RIGHT_SHOULDER], keypoints[RIGHT_ELBOW], keypoints[RIGHT_WRIST]
        ),
        "left_knee_angle": calculate_angle(
            keypoints[LEFT_HIP], keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE]
        ),
        "right_knee_angle": calculate_angle(
            keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE]
        ),
    }


KINEMATIC_SUMMARY_NAMES = [f"kin_summary_{i:02d}" for i in range(86)]


def extract_kinematic_summary_86d(seq_1140: np.ndarray) -> np.ndarray:
    """
    Day 03 정규화 관절 좌표 및 4대 관절 각도 시퀀스(30x38=1,140차원)에서
    동작 발생 시점(Shift) 및 좌우 방향(Flip)·카메라 각도에 불변인 86차원 물리 운동학 요약 특징을 추출합니다.
    """
    frames = np.asarray(seq_1140, dtype=np.float32).reshape((30, 38))
    coords = frames[:, :34].reshape((30, 17, 2))
    sh_mid = (coords[:, 5, :] + coords[:, 6, :]) / 2.0
    torso_s = np.maximum(np.linalg.norm(sh_mid, axis=1), 0.40)[:, None, None]
    tc = coords / torso_s

    l_elb, r_elb = frames[:, 34], frames[:, 35]
    l_knee, r_knee = frames[:, 36], frames[:, 37]
    avg_elb = (l_elb + r_elb) / 2.0
    min_elb_lr = np.minimum(l_elb, r_elb)
    max_elb_lr = np.maximum(l_elb, r_elb)
    avg_knee = (l_knee + r_knee) / 2.0
    min_knee_lr = np.minimum(l_knee, r_knee)
    max_knee_lr = np.maximum(l_knee, r_knee)

    sp = (tc[:, 5, :] + tc[:, 6, :]) / 2.0
    sp_tilt = np.degrees(np.arctan2(np.abs(sp[:, 0]), np.maximum(-sp[:, 1], 1e-3)))
    sh_y, sh_x = sp[:, 1], sp[:, 0]

    nose_x, nose_y = tc[:, 0, 0], tc[:, 0, 1]
    nose_rel_sh_y = nose_y - sh_y

    l_wx, l_wy = tc[:, 9, 0], tc[:, 9, 1]
    r_wx, r_wy = tc[:, 10, 0], tc[:, 10, 1]
    w_mid_x, w_mid_y = (l_wx + r_wx) / 2.0, (l_wy + r_wy) / 2.0
    w_dist = np.hypot(l_wx - r_wx, l_wy - r_wy)

    l_wy_sh, r_wy_sh = l_wy - sh_y, r_wy - sh_y
    w_rel_nose_y = w_mid_y - nose_y
    min_single_w_rel_nose = np.minimum(l_wy, r_wy) - nose_y

    act_wy_span = max(float(np.ptp(l_wy)), float(np.ptp(r_wy)))
    sec_wy_span = min(float(np.ptp(l_wy)), float(np.ptp(r_wy)))
    two_hand_y_ratio = sec_wy_span / max(act_wy_span, 1e-4)

    act_wx_span = max(float(np.ptp(l_wx)), float(np.ptp(r_wx)))
    sec_wx_span = min(float(np.ptp(l_wx)), float(np.ptp(r_wx)))
    two_hand_x_ratio = sec_wx_span / max(act_wx_span, 1e-4)

    wy_corr = float(np.corrcoef(l_wy, r_wy)[0, 1]) if (np.std(l_wy) > 1e-3 and np.std(r_wy) > 1e-3) else 0.0
    wx_corr = float(np.corrcoef(l_wx, r_wx)[0, 1]) if (np.std(l_wx) > 1e-3 and np.std(r_wx) > 1e-3) else 0.0

    l_kx, l_ky = tc[:, 13, 0], tc[:, 13, 1]
    r_kx, r_ky = tc[:, 14, 0], tc[:, 14, 1]
    l_ax, l_ay = tc[:, 15, 0], tc[:, 15, 1]
    r_ax, r_ay = tc[:, 16, 0], tc[:, 16, 1]
    ank_mid_y = (l_ay + r_ay) / 2.0
    knee_mid_y = (l_ky + r_ky) / 2.0
    ank_asym_y = np.abs(l_ay - r_ay)
    ank_sep_2d = np.hypot(l_ax - r_ax, l_ay - r_ay)
    knee_asym_y = np.abs(l_ky - r_ky)

    # 측면 낙상(fall_sideways) vs 전후방 낙상 구분용 좌우 롤(Roll) 비대칭 특징
    sh_roll_y = np.abs(tc[:, 5, 1] - tc[:, 6, 1])
    hip_roll_y = np.abs(tc[:, 11, 1] - tc[:, 12, 1])
    ear_roll_y = np.abs(tc[:, 3, 1] - tc[:, 4, 1])
    sh_width_tc = np.hypot(tc[:, 5, 0] - tc[:, 6, 0], tc[:, 5, 1] - tc[:, 6, 1])

    d_tc = np.diff(tc, axis=0)
    w_speed = np.maximum(np.linalg.norm(d_tc[:, 9, :], axis=1), np.linalg.norm(d_tc[:, 10, :], axis=1))
    a_speed = np.maximum(np.linalg.norm(d_tc[:, 15, :], axis=1), np.linalg.norm(d_tc[:, 16, :], axis=1))
    n_speed = np.linalg.norm(d_tc[:, 0, :], axis=1)
    d_elb = np.abs(np.diff(avg_elb))
    d_wy_mid = np.diff(w_mid_y)
    slam_3step = w_mid_y[3:] - w_mid_y[:-3]  # 양수 = 3스텝 동안 급격한 양손 하강(샷건!)

    peaks_elb, _ = find_peaks(avg_elb, prominence=12.0, height=95.0)
    valleys_elb, _ = find_peaks(-avg_elb, prominence=12.0)
    d_elb_signed = np.diff(avg_elb)
    elb_reversals = int(np.sum((d_elb_signed[:-1] * d_elb_signed[1:]) < -4.0))
    wy_reversals = int(np.sum((d_wy_mid[:-1] * d_wy_mid[1:]) < -0.002))

    sk_w = np.max(tc[:, :, 0], axis=1) - np.min(tc[:, :, 0], axis=1)
    sk_h = np.max(tc[:, :, 1], axis=1) - np.min(tc[:, :, 1], axis=1)
    wh_ratio = sk_w / np.maximum(sk_h, 0.2)

    s_feats = [
        # 1. 팔꿈치 각도 및 진동 (12)
        float(np.min(avg_elb)), float(np.max(avg_elb)), float(np.mean(avg_elb)), float(np.std(avg_elb)),
        float(np.min(min_elb_lr)), float(np.max(max_elb_lr)),
        float(np.sum(d_elb)), float(np.max(d_elb)),
        float(len(peaks_elb)), float(len(valleys_elb)), float(elb_reversals),
        float(np.sum(avg_elb > 110.0)),

        # 2. 무릎 각도 및 하체 비대칭 (14)
        float(np.min(min_knee_lr)), float(np.percentile(min_knee_lr, 10)), float(np.mean(avg_knee)), float(np.std(avg_knee)),
        float(np.median(max_knee_lr)), float(np.max(max_knee_lr - min_knee_lr)),
        float(np.min(ank_mid_y)), float(np.max(ank_mid_y)), float(np.mean(ank_mid_y)), float(np.ptp(ank_mid_y)),
        float(np.min(np.minimum(l_ay, r_ay))), float(np.max(ank_asym_y)), float(np.max(ank_sep_2d)), float(np.max(knee_asym_y)),

        # 3. 손목 궤적, 양손 대칭성, 머리 위 타격 충격량 (24)
        float(np.min(w_mid_y)), float(np.max(w_mid_y)), float(np.mean(w_mid_y)), float(np.ptp(w_mid_y)),
        float(np.ptp(w_mid_x)), float(np.std(w_mid_x)),
        act_wy_span, sec_wy_span, two_hand_y_ratio,
        act_wx_span, sec_wx_span, two_hand_x_ratio,
        wy_corr, wx_corr,
        float(np.min(np.minimum(l_wy_sh, r_wy_sh))), float(np.min(np.maximum(l_wy_sh, r_wy_sh))),
        float(np.max(np.maximum(l_wy, r_wy))), float(np.min(w_dist)), float(np.max(w_dist)),
        float(wy_reversals),
        float(np.min(w_rel_nose_y)), float(np.min(min_single_w_rel_nose)),
        float(np.max(slam_3step)), float(np.min(slam_3step)),

        # 4. 척추 기울기, 좌우 롤 비대칭, 종횡비 및 관절 속도 (26)
        float(np.min(nose_y)), float(np.max(nose_y)), float(np.mean(nose_y)), float(np.ptp(nose_y)),
        float(np.ptp(nose_x)), float(np.max(nose_rel_sh_y)),
        float(np.max(sp_tilt)), float(np.percentile(sp_tilt, 85)), float(np.mean(sp_tilt)),
        float(np.max(wh_ratio)), float(np.percentile(wh_ratio, 85)), float(np.min(wh_ratio)),
        float(np.max(w_speed)), float(np.mean(w_speed)),
        float(np.max(a_speed)), float(np.mean(a_speed)),
        float(np.max(n_speed)), float(np.max(d_wy_mid)),
        float(np.max(sh_roll_y)), float(np.mean(sh_roll_y)),
        float(np.max(hip_roll_y)), float(np.mean(hip_roll_y)),
        float(np.max(ear_roll_y)), float(np.min(sh_width_tc)),
        float(np.max(sh_width_tc)), float(np.ptp(sh_x)),

        # 5. 시작-중반-종료 자세 변화량 (10)
        float(np.mean(nose_y[10:20]) - np.mean(nose_y[:5])),
        float(np.mean(nose_y[24:]) - np.mean(nose_y[:5])),
        float(np.mean(ank_mid_y[10:20]) - np.mean(ank_mid_y[:5])),
        float(np.mean(ank_mid_y[24:]) - np.mean(ank_mid_y[:5])),
        float(np.mean(avg_knee[10:20]) - np.mean(avg_knee[:5])),
        float(np.mean(avg_knee[24:]) - np.mean(avg_knee[:5])),
        float(np.mean(sp_tilt[10:20]) - np.mean(sp_tilt[:5])),
        float(np.mean(sp_tilt[24:]) - np.mean(sp_tilt[:5])),
        float(np.mean(knee_mid_y[10:20])),
        float(np.max(np.abs(sh_x))),
    ]
    return np.nan_to_num(np.asarray(s_feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
