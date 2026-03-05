import cv2 as cv
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN=10
FONT_SIZE=1
FONT_THICKNESS=1
HANDEDNESS_TEXT_COLOR=(88, 205, 54)

def draw_hand_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

  return annotated_image

def draw_pose_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmark_style = mp_drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

    # Loop through the detected hands to visualize.
    for pose_landmarks in pose_landmarks_list:
        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style)

    return annotated_image

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# Create a hand landmarker instance with the video mode:
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=VisionRunningMode.VIDEO
)

timestamp = 0
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

            print(hand_res.handedness)
            for hand in hand_res.hand_landmarks:
                for r in hand:
                    print(r)

            annotated = draw_hand_landmarks_on_image(img, hand_res)
            annotated_ = draw_pose_landmarks_on_image(annotated, pose_res)
            cv.imshow('Hand Tracking', annotated_)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            timestamp += 1
    
cap.release()
cv.destroyAllWindows()
