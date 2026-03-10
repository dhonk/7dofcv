import math
from src.calculation.extract_angles import (
    calc_elbow_angle, calc_wrist_flex_angle, calc_wrist_rot_angle,
    calc_shoulder_horiz_angle, calc_shoulder_vert_angle,
)


def generate_elbow_angle(shoulder: tuple[float, float, float],
                         elbow: tuple[float, float, float],
                         wrist: tuple[float, float, float]) -> float:
    '''
    Compute robot elbow joint angle.
    Returns angle in radians: 0 = fully extended, π = fully flexed.
    '''
    angle = calc_elbow_angle(shoulder, elbow, wrist)
    return angle

def generate_wrist_flex_angle(elbow: tuple[float, float, float],
                              wrist: tuple[float, float, float],
                              index: tuple[float, float, float]) -> float:
    '''
    Compute robot wrist flexion joint angle.
    Returns angle in radians
    '''
    angle = calc_wrist_flex_angle(elbow, wrist, index)
    return angle

def generate_wrist_rot_angle(index_mcp: tuple[float, float, float],
                             pinky_mcp: tuple[float, float, float],
                             wrist: tuple[float, float, float]) -> float:
    angle = calc_wrist_rot_angle(wrist, index_mcp, pinky_mcp)
    print(f"[WRIST_ROT] index_mcp=({index_mcp[0]:.4f}, {index_mcp[1]:.4f}, {index_mcp[2]:.4f}) "
    f"wrist_hand=({wrist[0]:.4f}, {wrist[1]:.4f}, {wrist[2]:.4f}) "
    f"pinky_mcp=({pinky_mcp[0]:.4f}, {pinky_mcp[1]:.4f}, {pinky_mcp[2]:.4f}) "
    f"angle={math.degrees(angle):.1f}°  ({angle:.4f} rad)")

    return angle

def generate_shoulder_rot(shoulder: tuple[float, float, float],
                          elbow: tuple[float, float, float],
                          wrist: tuple[float, float, float]) -> float:
    angle = 0
    print(f"[SHOULDER_ROT] shoulder=({shoulder[0]:.4f}, {shoulder[1]:.4f}, {shoulder[2]:.4f}) "
    f"elbow=({elbow[0]:.4f}, {elbow[1]:.4f}, {elbow[2]:.4f}) "
    f"wrist=({wrist[0]:.4f}, {wrist[1]:.4f}, {wrist[2]:.4f}) "
    f"angle={math.degrees(angle):.1f}°  ({angle:.4f} rad)")

    return angle

def generate_shoulder_horiz(opp_shoulder: tuple[float, float, float],
                           shoulder: tuple[float, float, float],
                           elbow: tuple[float, float, float]) -> float:
    angle = 0
    print(f"[SHOULDER_HORIZ] opp_shoulder=({opp_shoulder[0]:.4f}, {opp_shoulder[1]:.4f}, {opp_shoulder[2]:.4f}) "
    f"shoulder=({shoulder[0]:.4f}, {shoulder[1]:.4f}, {shoulder[2]:.4f}) "
    f"elbow=({elbow[0]:.4f}, {elbow[1]:.4f}, {elbow[2]:.4f}) "
    f"angle={math.degrees(angle):.1f}°  ({angle:.4f} rad)")

    return angle

def generate_shoulder_vert(opp_shoulder: tuple[float, float, float],
                           shoulder: tuple[float, float, float],
                           elbow: tuple[float, float, float]) -> float:
    angle = 0
    print(f"[SHOULDER_VERT] opp_shoulder=({opp_shoulder[0]:.4f}, {opp_shoulder[1]:.4f}, {opp_shoulder[2]:.4f}) "
    f"shoulder=({shoulder[0]:.4f}, {shoulder[1]:.4f}, {shoulder[2]:.4f}) "
    f"elbow=({elbow[0]:.4f}, {elbow[1]:.4f}, {elbow[2]:.4f}) "
    f"angle={math.degrees(angle):.1f}°  ({angle:.4f} rad)")

    return angle
