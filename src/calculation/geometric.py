import math
import numpy as np


def generate_elbow_angle(shoulder: tuple[float, float, float],
                         elbow: tuple[float, float, float],
                         wrist: tuple[float, float, float]) -> float:
    '''
    Compute elbow flexion angle.
    Returns angle in radians: 0 = fully extended, π = fully flexed.
    '''
    e = np.array(elbow[:2])
    v1 = np.array(shoulder[:2]) - e  # upper arm
    v2 = np.array(wrist[:2]) - e     # forearm

    cross = float(np.cross(v1, v2))
    dot = float(np.dot(v1, v2))
    angle = math.pi - math.atan2(abs(cross), dot)

    print(f"[ELBOW] shoulder=({shoulder[0]:.4f}, {shoulder[1]:.4f}, {shoulder[2]:.4f}) "
          f"elbow=({elbow[0]:.4f}, {elbow[1]:.4f}, {elbow[2]:.4f}) "
          f"wrist=({wrist[0]:.4f}, {wrist[1]:.4f}, {wrist[2]:.4f}) "
          f"angle={math.degrees(angle):.1f}°  ({angle:.4f} rad)")

    return angle
