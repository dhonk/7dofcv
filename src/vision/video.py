import csv
import os
from datetime import datetime
import cv2 as cv
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path='hand_landmarker.task',
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path='pose_landmarker_full.task',
    ),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.3,
)


def draw_hand_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv.putText(annotated_image, f"{handedness[0].category_name}",
                   (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                   FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

    return annotated_image


ArmRecord = dict  # {frame, shoulder_x/y/z, elbow_x/y/z, wrist_x/y/z}

_RIGHT_SHOULDER = 12
_RIGHT_ELBOW    = 14
_RIGHT_WRIST    = 16

CSV_PATH = f"arm_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
CSV_FIELDS = [
    "frame",
    "shoulder_x", "shoulder_y", "shoulder_z",
    "elbow_x",    "elbow_y",    "elbow_z",
    "wrist_x",    "wrist_y",    "wrist_z",
    "human_elbow_angle",
    "robot_joint3_angle",
]


def extract_arm_landmarks(detection_result, frame: int) -> ArmRecord | None:
    """Return x/y/z for right shoulder, elbow, wrist; None if no pose detected."""
    landmarks = detection_result.pose_landmarks
    if not landmarks:
        return None
    lm = landmarks[0]
    s, e, w = lm[_RIGHT_SHOULDER], lm[_RIGHT_ELBOW], lm[_RIGHT_WRIST]
    return {
        "frame":      frame,
        "shoulder_x": s.x, "shoulder_y": s.y, "shoulder_z": s.z,
        "elbow_x":    e.x, "elbow_y":    e.y, "elbow_z":    e.z,
        "wrist_x":    w.x, "wrist_y":    w.y, "wrist_z":    w.z,
    }


def save_arm_csv(records: list[ArmRecord], path: str = CSV_PATH) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} frames to {path}")


def plot_arm_tracking(records: list[ArmRecord]) -> None:
    if not records:
        return
    frames = [r["frame"] for r in records]
    joints = {
        "Right Shoulder": ("shoulder_x", "shoulder_y", "shoulder_z"),
        "Right Elbow":    ("elbow_x",    "elbow_y",    "elbow_z"),
        "Right Wrist":    ("wrist_x",    "wrist_y",    "wrist_z"),
    }
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, (joint, (xk, yk, zk)) in zip(axes, joints.items()):
        ax.plot(frames, [r[xk] for r in records], label="x")
        ax.plot(frames, [r[yk] for r in records], label="y")
        ax.plot(frames, [r[zk] for r in records], label="z")
        ax.set_title(joint)
        ax.set_ylabel("Position (normalized)")
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    plt.show()


def draw_pose_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmark_style = mp_drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_landmarks_list:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style)

    return annotated_image
