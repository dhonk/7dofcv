import logging
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

logger = logging.getLogger(__name__)

JOINT_NAMES: list[str] = [
    "/Franka/joint",
    "/Franka/link2_resp/joint",
    "/Franka/link3_resp/joint",
    "/Franka/link4_resp/joint",
    "/Franka/link5_resp/joint",
    "/Franka/link6_resp/joint",
    "/Franka/link7_resp/joint",
]


class FrankaPanda:
    def __init__(self, host: str = "172.21.80.1", port: int = 23000) -> None:
        logger.debug("Connecting to CoppeliaSim at %s:%d", host, port)
        self._client = RemoteAPIClient(host=host, port=port)
        self._sim = self._client.require("sim")
        self._joints: list[int] = [
            self._sim.getObject(name) for name in JOINT_NAMES
        ]
        for handle in self._joints:
            self._sim.setJointMode(handle, self._sim.jointmode_kinematic, 0)
        logger.debug("Connected — %d joints initialized", len(self._joints))

    def set_joint_angles(self, angles: list[float]) -> None:
        """Send target joint angles (in radians) to all 7 joints."""
        if len(angles) != 7:
            raise ValueError(f"Expected 7 joint angles, got {len(angles)}")
        logger.debug("set_joint_angles: %s", [f"{a:.4f}" for a in angles])
        for handle, angle in zip(self._joints, angles):
            self._sim.setJointPosition(handle, angle)

    def get_joint_angles(self) -> list[float]:
        """Read current joint positions (in radians) for all 7 joints."""
        angles = [self._sim.getJointPosition(handle) for handle in self._joints]
        logger.debug("get_joint_angles: %s", [f"{a:.4f}" for a in angles])
        return angles

    def get_joint_forces(self) -> list[float]:
        """Read the force/torque applied on each joint along its active axis."""
        return [self._sim.getJointForce(handle) for handle in self._joints]

    def get_joint_vel(self) -> list[float]:
        """Read the current velocity of each joint (rad/s for revolute joints)."""
        return [self._sim.getJointVelocity(handle) for handle in self._joints]

    def get_joint_positions_3d(self) -> list[list[float]]:
        """Return world-space [x, y, z] for each of the 7 joints."""
        return [self._sim.getObjectPosition(handle, -1) for handle in self._joints]

    def get_eef_pose(self) -> tuple[list[float], list[float]]:
        """Return ([x,y,z], [alpha,beta,gamma]) of the end-effector (/Franka/connection)."""
        handle = self._sim.getObject("/Franka/connection")
        pos = self._sim.getObjectPosition(handle, -1)
        ori = self._sim.getObjectOrientation(handle, -1)
        return pos, ori
