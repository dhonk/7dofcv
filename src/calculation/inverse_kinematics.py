import os
import math
from ikpy.chain import Chain

# Franka Panda maximum reach (metres), used to scale normalised IK targets
ROBOT_ARM_REACH_M = 0.855

_URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "franka.urdf")
_chain = Chain.from_urdf_file(
    os.path.abspath(_URDF_PATH),
    base_elements=["robot_base"],
    active_links_mask=[False, True, True, True, True, True, True, True],
)

def solve_ik(target_xyz: tuple, initial_angles: list | None = None) -> list[float]:
    """
    Compute joint angles for the Franka Panda to reach target_xyz.

    Args:
        target_xyz: (x, y, z) end-effector target position in metres.
        initial_angles: Optional seed of 7 joint angles (radians). Defaults to zeros.

    Returns:
        List of 7 joint angles in radians (joints 1–7).
    """
    if initial_angles is None:
        initial_position = [0.0] * len(_chain.links)
    else:
        initial_position = [0.0] + list(initial_angles)

    result = _chain.inverse_kinematics(target_xyz, initial_position=initial_position)
    return list(result[1:8])

def get_arm_reach() -> float:
    return ROBOT_ARM_REACH_M

def get_home_position() -> tuple[float, float, float]:
    """End-effector (x,y,z) in robot_base frame at all-zero joint angles."""
    fk = _chain.forward_kinematics([0.0] * len(_chain.links))
    return (fk[0, 3], fk[1, 3], fk[2, 3])
