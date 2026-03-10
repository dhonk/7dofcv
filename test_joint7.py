import time
import math
from src.robot.coppelia import FrankaPanda

robot = FrankaPanda()

for deg in [0, 90, 180]:
    rad = math.radians(deg)
    joints = robot.get_joint_angles()
    joints[6] = rad
    robot.set_joint_angles(joints)
    print(f"Joint 7 set to {deg}° ({rad:.4f} rad)")
    time.sleep(3)

print("Done")
