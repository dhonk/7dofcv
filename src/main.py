import argparse
import cv2 as cv

import sys
sys.path.insert(0, ".")

from src.vision.video import (
    LatestFrameCapture, downscale_frame,
    draw_hand_landmarks_on_image, draw_pose_landmarks_on_image,
)
from src.vision.process import (
    HandLandmarker, PoseLandmarker,
    hand_options, pose_options,
    create_mp_image, extract_arm_landmarks, extract_hand_landmarks,
    extract_pose_tuples, extract_hand_tuples, extract_full_pose_tuples,
)
from src.calculation.geometric import (
    generate_elbow_angle, generate_wrist_flex_angle, generate_wrist_rot_angle,
    generate_shoulder_horiz, generate_shoulder_vert,
)
from src.robot.coppelia import FrankaPanda
from src.data_processing.save_data import save_arm_csv, plot_arm_tracking

parser = argparse.ArgumentParser(description="Arm tracking with robot control")
parser.add_argument("--joint", nargs="+", default=["all"],
                    help="Robot joint(s) to drive (1-7 or 'all'). Default: all")
args = parser.parse_args()
if "all" in args.joint:
    active_joints = set(range(1, 8))
else:
    active_joints = {int(j) for j in args.joint}

print("[INFO] Opening video capture...")
cap = LatestFrameCapture('udp://0.0.0.0:5005?overrun_nonfatal=1')
print("[INFO] Video capture opened")

print("[INFO] Connecting to robot...")
robot = FrankaPanda()
print("[INFO] Robot connected")

timestamp = 0
arm_records = []

print("[INFO] Creating landmarkers...")
with HandLandmarker.create_from_options(hand_options) as h_landmarker:
    with PoseLandmarker.create_from_options(pose_options) as p_landmarker:
        print("[INFO] Landmarkers ready, entering main loop...")
        while True:
            ret, img = cap.read()
            if not ret or img is None:
                continue
 
            img = downscale_frame(img)
            mp_image = create_mp_image(img)
            hand_res = h_landmarker.detect_for_video(mp_image, timestamp)
            pose_res = p_landmarker.detect_for_video(mp_image, timestamp)

            joints = robot.get_joint_angles()

            record = extract_arm_landmarks(pose_res, timestamp)
            if record is not None and timestamp % 10:
                hand_record = extract_hand_landmarks(hand_res)
                if hand_record:
                    record.update(hand_record)

                pose_tuples = extract_pose_tuples(pose_res)
                shoulder, elbow, wrist, index = pose_tuples

                full_pose_tuples = extract_full_pose_tuples(pose_res)
                opp_shoulder, shoulder_full, elbow_full, _wrist_full, _index_full = full_pose_tuples

                hand_tuples = extract_hand_tuples(hand_res)

                shoulder_horiz_angle = 0.0
                shoulder_vertz_angle = 0.0
                elbow_angle = 0.0
                wrist_rot_angle = 0.0
                wrist_flex_angle = 0.0

                if 1 in active_joints:
                    shoulder_horiz_angle = generate_shoulder_horiz(opp_shoulder, shoulder_full, elbow_full)
                    record["human_shoulder_horiz_angle"] = shoulder_horiz_angle
                    record["robot_joint1_angle"] = joints[0]

                if 2 in active_joints:
                    shoulder_vert_angle = generate_shoulder_vert(opp_shoulder, shoulder_full, elbow_full)
                    record["human_shoulder_vert_angle"] = shoulder_vert_angle
                    record["robot_joint2_angle"] = joints[1]

                if 4 in active_joints:
                    elbow_angle = generate_elbow_angle(shoulder, elbow, wrist)
                    record["human_elbow_angle"] = elbow_angle
                    record["robot_joint4_angle"] = joints[3]

                if 5 in active_joints:
                    if hand_tuples:
                        wrist_hand, index_mcp, pinky_mcp = hand_tuples
                        wrist_rot_angle = generate_wrist_rot_angle(index_mcp, pinky_mcp, wrist_hand)
                    record["human_wrist_rot_angle"] = wrist_rot_angle
                    record["robot_joint5_angle"] = joints[4]

                if 6 in active_joints:
                    wrist_flex_angle = generate_wrist_flex_angle(elbow, wrist, index)
                    record["human_wrist_flex_angle"] = wrist_flex_angle
                    record["robot_joint6_angle"] = joints[5]

                joints[0] = shoulder_horiz_angle if 1 in active_joints else 0
                joints[1] = shoulder_vert_angle if 2 in active_joints else 0
                joints[2] = 0
                joints[3] = -elbow_angle if 4 in active_joints else 0
                joints[4] = wrist_rot_angle if 5 in active_joints else 0
                joints[5] = wrist_flex_angle if 6 in active_joints else 0
                joints[6] = 0

                for i, angle in enumerate(joints):
                    record[f"sent_joint{i + 1}"] = angle

                arm_records.append(record)
                robot.set_joint_angles(joints)

            draw_hand_landmarks_on_image(img, hand_res)
            draw_pose_landmarks_on_image(img, pose_res, joints)
            cv.imshow('Hand and Pose Tracking', img)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            timestamp += 1

cap.release()
cv.destroyAllWindows()

save_arm_csv(arm_records)
plot_arm_tracking(arm_records, active_joints)
