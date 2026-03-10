import math
import cv2 as cv
import mediapipe as mp
import threading

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)


class LatestFrameCapture:
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


def draw_pose_landmarks_on_image(rgb_image, detection_result, joints):
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

        height, width, _ = rgb_image.shape

        # Shoulder label (landmark 12 = right shoulder)
        shoulder_text_x = int(pose_landmarks[12].x * width)
        shoulder_text_y = int(pose_landmarks[12].y * height) - MARGIN
        cv.putText(rgb_image, f"{math.degrees(joints[1]):.1f}deg",
                   (shoulder_text_x, shoulder_text_y), cv.FONT_HERSHEY_DUPLEX,
                   FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

        # Elbow label (landmark 14 = right elbow)
        elbow_text_x = int(pose_landmarks[14].x * width)
        elbow_text_y = int(pose_landmarks[14].y * height) - MARGIN
        cv.putText(rgb_image, f"{math.degrees(joints[3]):.1f}deg",
                   (elbow_text_x, elbow_text_y), cv.FONT_HERSHEY_DUPLEX,
                   FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

        # Wrist label (landmark 16 = right wrist)
        wrist_text_x = int(pose_landmarks[16].x * width)
        wrist_text_y = int(pose_landmarks[16].y * height) - MARGIN
        cv.putText(rgb_image, f"{math.degrees(joints[5]):.1f}deg",
                   (wrist_text_x, wrist_text_y), cv.FONT_HERSHEY_DUPLEX,
                   FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)
