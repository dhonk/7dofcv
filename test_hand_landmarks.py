import cv2 as cv

from src.vision.video import LatestFrameCapture, downscale_frame, draw_hand_landmarks_on_image
from src.vision.process import HandLandmarker, hand_options, create_mp_image

cap = LatestFrameCapture('udp://0.0.0.0:5005?overrun_nonfatal=1')

timestamp = 0

with HandLandmarker.create_from_options(hand_options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = downscale_frame(frame)
        mp_image = create_mp_image(frame)

        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks and timestamp % 10 == 0:
            lm = result.hand_landmarks[0]
            wrist     = lm[0]
            index_mcp = lm[5]
            pinky_mcp = lm[17]
            print(f"[{timestamp}] "
                  f"wrist=({wrist.x:.4f}, {wrist.y:.4f}, {wrist.z:.4f})  "
                  f"index_mcp=({index_mcp.x:.4f}, {index_mcp.y:.4f}, {index_mcp.z:.4f})  "
                  f"pinky_mcp=({pinky_mcp.x:.4f}, {pinky_mcp.y:.4f}, {pinky_mcp.z:.4f})")

        draw_hand_landmarks_on_image(frame, result)
        cv.imshow('Hand Landmarks', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        timestamp += 1

cap.release()
cv.destroyAllWindows()
