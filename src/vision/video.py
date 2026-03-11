import math
import cv2 as cv
import mediapipe as mp
import threading
_RIGHT_SHOULDER = 12
_RIGHT_ELBOW    = 14
_RIGHT_WRIST    = 16

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
LABEL_FONT_SIZE = 0.45
LABEL_FONT_THICKNESS = 1
THETA_COLORS = [
    (255,  80,  80),  # t1 — red
    (255, 160,  40),  # t2 — orange
    (255, 230,  40),  # t3 — yellow
    ( 80, 220,  80),  # t4 — green
    ( 40, 200, 255),  # t5 — cyan
    ( 80,  80, 255),  # t6 — blue
    (200,  80, 255),  # t7 — violet
]


class LatestFrameCaptureLive:
    """Threaded capture to always grab the latest frame and drain the UDP buffer."""
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


def downscale_frame(frame, max_width=640):
    """Downscale frame if wider than max_width, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv.resize(frame, (max_width, int(h * scale)))
    return frame


def draw_hand_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        mp_drawing.draw_landmarks(
            rgb_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        height, width, _ = rgb_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv.putText(rgb_image, f"{handedness[0].category_name}",
                   (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                   FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)


def _draw_theta1_label(rgb_image, pose_landmarks, angle_rad):
    height, width, _ = rgb_image.shape
    lm = pose_landmarks[_RIGHT_SHOULDER]
    cv.putText(rgb_image, f"t1 shoulder rot:{math.degrees(angle_rad):.1f}deg",
               (int(lm.x * width), int(lm.y * height) - MARGIN),
               cv.FONT_HERSHEY_DUPLEX, LABEL_FONT_SIZE, THETA_COLORS[0], LABEL_FONT_THICKNESS, cv.LINE_AA)

def _draw_theta2_label(rgb_image, pose_landmarks, angle_rad):
    height, width, _ = rgb_image.shape
    lm = pose_landmarks[_RIGHT_SHOULDER]
    cv.putText(rgb_image, f"t2 shoulder flex:{math.degrees(angle_rad):.1f}deg",
               (int(lm.x * width), int(lm.y * height) - MARGIN * 3),
               cv.FONT_HERSHEY_DUPLEX, LABEL_FONT_SIZE, THETA_COLORS[1], LABEL_FONT_THICKNESS, cv.LINE_AA)

def _draw_theta3_label(rgb_image, pose_landmarks, angle_rad):
    height, width, _ = rgb_image.shape
    lm = pose_landmarks[_RIGHT_SHOULDER]
    cv.putText(rgb_image, f"t3 arm rot:{math.degrees(angle_rad):.1f}deg",
               (int(lm.x * width), int(lm.y * height) - MARGIN * 5),
               cv.FONT_HERSHEY_DUPLEX, LABEL_FONT_SIZE, THETA_COLORS[2], LABEL_FONT_THICKNESS, cv.LINE_AA)

def _draw_theta4_label(rgb_image, pose_landmarks, angle_rad):
    height, width, _ = rgb_image.shape
    lm = pose_landmarks[_RIGHT_ELBOW]
    cv.putText(rgb_image, f"t4 elbow flex:{math.degrees(angle_rad):.1f}deg",
               (int(lm.x * width), int(lm.y * height) - MARGIN),
               cv.FONT_HERSHEY_DUPLEX, LABEL_FONT_SIZE, THETA_COLORS[3], LABEL_FONT_THICKNESS, cv.LINE_AA)

def _draw_theta5_label(rgb_image, pose_landmarks, angle_rad):
    height, width, _ = rgb_image.shape
    lm = pose_landmarks[_RIGHT_ELBOW]
    cv.putText(rgb_image, f"t5 wrist rot:{math.degrees(angle_rad):.1f}deg",
               (int(lm.x * width), int(lm.y * height) - MARGIN * 3),
               cv.FONT_HERSHEY_DUPLEX, LABEL_FONT_SIZE, THETA_COLORS[4], LABEL_FONT_THICKNESS, cv.LINE_AA)

def _draw_theta6_label(rgb_image, pose_landmarks, angle_rad):
    height, width, _ = rgb_image.shape
    lm = pose_landmarks[_RIGHT_WRIST]
    cv.putText(rgb_image, f"t6 wrist flex:{math.degrees(angle_rad):.1f}deg",
               (int(lm.x * width), int(lm.y * height) - MARGIN),
               cv.FONT_HERSHEY_DUPLEX, LABEL_FONT_SIZE, THETA_COLORS[5], LABEL_FONT_THICKNESS, cv.LINE_AA)

def _draw_theta7_label(rgb_image, pose_landmarks, angle_rad):
    height, width, _ = rgb_image.shape
    lm = pose_landmarks[_RIGHT_WRIST]
    cv.putText(rgb_image, f"t7 wrist flex (stub):{math.degrees(angle_rad):.1f}deg",
               (int(lm.x * width), int(lm.y * height) - MARGIN * 3),
               cv.FONT_HERSHEY_DUPLEX, LABEL_FONT_SIZE, THETA_COLORS[6], LABEL_FONT_THICKNESS, cv.LINE_AA)


def draw_pose_landmarks_on_image(rgb_image, detection_result, joints, active_joints=None, inverse=False):
    if active_joints is None:
        active_joints = set(range(1, 8))

    pose_landmarks_list = detection_result.pose_landmarks

    pose_landmark_style = mp_drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_landmarks_list:
        mp_drawing.draw_landmarks(
            image=rgb_image,
            landmark_list=pose_landmarks,
            connections=mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style)

        if inverse:
            height, width, _ = rgb_image.shape
            wrist = pose_landmarks[_RIGHT_WRIST]
            shoulder = pose_landmarks[_RIGHT_SHOULDER]
            dx = wrist.x - shoulder.x
            dy = wrist.y - shoulder.y
            dz = wrist.z - shoulder.z
            px = int(wrist.x * width)
            py = int(wrist.y * height)
            label = f"x:{dx:.3f} y:{dy:.3f} z:{dz:.3f}"
            cv.putText(rgb_image, label, (px + MARGIN, py - MARGIN),
                       cv.FONT_HERSHEY_DUPLEX, LABEL_FONT_SIZE, (255, 255, 255),
                       LABEL_FONT_THICKNESS, cv.LINE_AA)
        else:
            if 1 in active_joints: _draw_theta1_label(rgb_image, pose_landmarks, joints[0])
            if 2 in active_joints: _draw_theta2_label(rgb_image, pose_landmarks, joints[1])
            if 3 in active_joints: _draw_theta3_label(rgb_image, pose_landmarks, joints[2])
            if 4 in active_joints: _draw_theta4_label(rgb_image, pose_landmarks, joints[3])
            if 5 in active_joints: _draw_theta5_label(rgb_image, pose_landmarks, joints[4])
            if 6 in active_joints: _draw_theta6_label(rgb_image, pose_landmarks, joints[5])
            if 7 in active_joints: _draw_theta7_label(rgb_image, pose_landmarks, joints[6])
