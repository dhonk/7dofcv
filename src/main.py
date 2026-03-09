import cv2 as cv
import mediapipe as mp
import threading

from src.vision.video import (
    hand_options, pose_options,
    draw_hand_landmarks_on_image, draw_pose_landmarks_on_image,
    extract_arm_landmarks, save_arm_csv, plot_arm_tracking,
)
from src.calculation.geometric import generate_elbow_angle
from src.robot.coppelia import FrankaPanda

HandLandmarker = mp.tasks.vision.HandLandmarker
PoseLandmarker = mp.tasks.vision.PoseLandmarker

# Threaded capture to always grab the latest frame and drain the UDP buffer
class LatestFrameCapture:
    def __init__(self, src):
        self.cap = cv.VideoCapture(src, cv.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video source")
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def release(self):
        self.stopped = True
        self.thread.join(timeout=2)
        self.cap.release()


PROCESS_WIDTH = 640

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

            # Downscale for faster processing
            h, w = img.shape[:2]
            if w > PROCESS_WIDTH:
                scale = PROCESS_WIDTH / w
                img = cv.resize(img, (PROCESS_WIDTH, int(h * scale)))

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

            draw_hand_landmarks_on_image(img, hand_res)
            draw_pose_landmarks_on_image(img, pose_res)
            cv.imshow('Hand Tracking', img)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            timestamp += 1

cap.release()
cv.destroyAllWindows()

save_arm_csv(arm_records)
plot_arm_tracking(arm_records)
