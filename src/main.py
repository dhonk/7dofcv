import cv2 as cv
import mediapipe as mp

from src.vision.video import (
    hand_options, pose_options,
    draw_hand_landmarks_on_image, draw_pose_landmarks_on_image,
    extract_arm_landmarks, save_arm_csv, plot_arm_tracking,
)
from src.vision.processing import generate_elbow_angle
from src.robot.coppelia import FrankaPanda

HandLandmarker = mp.tasks.vision.HandLandmarker
PoseLandmarker = mp.tasks.vision.PoseLandmarker

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

robot = FrankaPanda()

timestamp = 0
arm_records = []

with HandLandmarker.create_from_options(hand_options) as h_landmarker:
    with PoseLandmarker.create_from_options(pose_options) as p_landmarker:
        while True:
            ret, img = cap.read()
            if not ret:
                print('empty camera frame')
                raise ValueError

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            hand_res = h_landmarker.detect_for_video(mp_image, timestamp)
            pose_res = p_landmarker.detect_for_video(mp_image, timestamp)

            record = extract_arm_landmarks(pose_res, timestamp)
            if record is not None and timestamp % 10:
                lm = pose_res.pose_landmarks[0]
                shoulder = (lm[12].x, lm[12].y, lm[12].z)
                elbow    = (lm[14].x, lm[14].y, lm[14].z)
                wrist    = (lm[16].x, lm[16].y, lm[16].z)
                elbow_angle = generate_elbow_angle(shoulder, elbow, wrist)
                joints = robot.get_joint_angles()
                record["human_elbow_angle"] = elbow_angle
                record["robot_joint3_angle"] = joints[3]
                arm_records.append(record)
                joints[0] = 0
                joints[1] = 0
                joints[2] = 0
                joints[3] = -elbow_angle
                joints[4] = 0
                joints[5] = 0
                joints[6] = 0
                
                print(elbow_angle)
                robot.set_joint_angles(joints)

            annotated = draw_hand_landmarks_on_image(img, hand_res)
            annotated_ = draw_pose_landmarks_on_image(annotated, pose_res)
            cv.imshow('Hand Tracking', annotated_)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            timestamp += 1

cap.release()
cv.destroyAllWindows()

save_arm_csv(arm_records)
plot_arm_tracking(arm_records)
