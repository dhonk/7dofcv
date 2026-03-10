import math
import numpy as np


def _vertex_angle_2d(p1, vertex, p2):
    """Angle at `vertex` formed by rays to p1 and p2, using x,y only.
    Returns atan2(|cross|, dot) in radians [0, π]."""
    v = np.array(vertex[:2])
    v1 = np.array(p1[:2]) - v
    v2 = np.array(p2[:2]) - v
    cross = float(np.cross(v1, v2))
    dot = float(np.dot(v1, v2))
    return math.atan2(abs(cross), dot)


def calc_elbow_angle(shoulder, elbow, wrist) -> float:
    """Compute elbow flexion angle.
    Returns angle in radians: 0 = fully extended, π = fully flexed."""
    angle = math.pi - _vertex_angle_2d(shoulder, elbow, wrist)
    print(f"[ELBOW] shoulder=({shoulder[0]:.4f}, {shoulder[1]:.4f}, {shoulder[2]:.4f}) "
          f"elbow=({elbow[0]:.4f}, {elbow[1]:.4f}, {elbow[2]:.4f}) "
          f"wrist=({wrist[0]:.4f}, {wrist[1]:.4f}, {wrist[2]:.4f}) "
          f"angle={math.degrees(angle):.1f}°  ({angle:.4f} rad)")
    return angle


def calc_wrist_flex_angle(elbow, wrist, index) -> float:
    """Compute wrist flexion angle. Returns angle in radians."""
    angle = _vertex_angle_2d(elbow, wrist, index)
    print(f"[WRIST_FLEX] elbow=({elbow[0]:.4f}, {elbow[1]:.4f}, {elbow[2]:.4f}) "
          f"wrist=({wrist[0]:.4f}, {wrist[1]:.4f}, {wrist[2]:.4f}) "
          f"index=({index[0]:.4f}, {index[1]:.4f}, {index[2]:.4f}) "
          f"angle={math.degrees(angle):.1f}°  ({angle:.4f} rad)")
    return angle


def calc_wrist_rot_angle(wrist, index_mcp, pinky_mcp) -> float:
    """Compute wrist rotation angle. Stub — returns 0.0."""
    return 0.0


def calc_shoulder_horiz_angle(opp_shoulder, shoulder, elbow) -> float:
    """Shoulder horizontal adduction angle.
    Positive when elbow moves towards body center.
    0 = arm hanging straight down, π/2 = arm horizontal towards center.
    Measures angle from 'straight down' rotating towards the center axis."""
    s = np.array(shoulder[:2])
    arm = np.array(elbow[:2]) - s

    # Local frame: h_hat towards center, down perpendicular to shoulder line
    h_axis = np.array(opp_shoulder[:2]) - s
    h_hat = h_axis / np.linalg.norm(h_axis)
    down = np.array([-h_hat[1], h_hat[0]])  # 90° CCW = down in mediapipe

    h_comp = float(np.dot(arm, h_hat))
    d_comp = float(np.dot(arm, down))
    angle = math.atan2(h_comp, d_comp)

    print(f"[SHOULDER_HORIZ] opp_shoulder=({opp_shoulder[0]:.4f}, {opp_shoulder[1]:.4f}, {opp_shoulder[2]:.4f}) "
          f"shoulder=({shoulder[0]:.4f}, {shoulder[1]:.4f}, {shoulder[2]:.4f}) "
          f"elbow=({elbow[0]:.4f}, {elbow[1]:.4f}, {elbow[2]:.4f}) "
          f"angle={math.degrees(angle):.1f}°  ({angle:.4f} rad)")
    return angle


def calc_shoulder_vert_angle(opp_shoulder, shoulder, elbow) -> float:
    """Shoulder vertical elevation angle.
    Positive when elbow moves towards head.
    0 = arm along shoulder line, π/2 = arm straight up.
    Measures angle from the shoulder line rotating upward."""
    s = np.array(shoulder[:2])
    arm = np.array(elbow[:2]) - s

    # Local frame: h_hat towards center, up perpendicular towards head
    h_axis = np.array(opp_shoulder[:2]) - s
    h_hat = h_axis / np.linalg.norm(h_axis)
    up = np.array([h_hat[1], -h_hat[0]])  # 90° CW = up in mediapipe (y-inverted)

    h_comp = float(np.dot(arm, h_hat))
    u_comp = float(np.dot(arm, up))
    angle = math.atan2(u_comp, h_comp)

    print(f"[SHOULDER_VERT] opp_shoulder=({opp_shoulder[0]:.4f}, {opp_shoulder[1]:.4f}, {opp_shoulder[2]:.4f}) "
          f"shoulder=({shoulder[0]:.4f}, {shoulder[1]:.4f}, {shoulder[2]:.4f}) "
          f"elbow=({elbow[0]:.4f}, {elbow[1]:.4f}, {elbow[2]:.4f}) "
          f"angle={math.degrees(angle):.1f}°  ({angle:.4f} rad)")
    return angle
