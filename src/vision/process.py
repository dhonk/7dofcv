import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path='./mediapipe-models/hand_landmarker.task',
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path='./mediapipe-models/pose_landmarker_full.task',
    ),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.3,
)

# Pose landmark indices
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_RIGHT_ELBOW    = 14
_RIGHT_WRIST    = 16
_RIGHT_INDEX    = 20

# Hand landmark indices
_HAND_WRIST     = 0
_HAND_INDEX_MCP = 5
_HAND_PINKY_MCP = 17
_HAND_RING_MCP  = 13


def create_mp_image(frame):
    """Wrap a BGR/RGB numpy frame into a mediapipe Image."""
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


def extract_arm_landmarks(detection_result, frame: int) -> dict | None:
    """Return x/y/z for right shoulder, elbow, wrist, index; None if no pose detected."""
    landmarks = detection_result.pose_landmarks
    if not landmarks:
        return None
    lm = landmarks[0]
    s, e, w, i = lm[_RIGHT_SHOULDER], lm[_RIGHT_ELBOW], lm[_RIGHT_WRIST], lm[_RIGHT_INDEX]
    return {
        "frame":      frame,
        "shoulder_x": s.x, "shoulder_y": s.y, "shoulder_z": s.z,
        "elbow_x":    e.x, "elbow_y":    e.y, "elbow_z":    e.z,
        "wrist_x":    w.x, "wrist_y":    w.y, "wrist_z":    w.z,
        "index_x":    i.x, "index_y":    i.y, "index_z":    i.z,
    }


def extract_hand_landmarks(detection_result) -> dict | None:
    """Return x/y/z for hand landmarks 0, 5, 13; None if no hand detected."""
    landmarks = detection_result.hand_landmarks
    if not landmarks:
        return None
    lm = landmarks[0]
    hw, im, rm = lm[_HAND_WRIST], lm[_HAND_INDEX_MCP], lm[_HAND_RING_MCP]
    return {
        "hand_wrist_x": hw.x, "hand_wrist_y": hw.y, "hand_wrist_z": hw.z,
        "hand_index_mcp_x": im.x, "hand_index_mcp_y": im.y, "hand_index_mcp_z": im.z,
        "hand_ring_mcp_x":  rm.x, "hand_ring_mcp_y":  rm.y, "hand_ring_mcp_z":  rm.z,
    }


def extract_pose_tuples(pose_result):
    """Return (shoulder, elbow, wrist, index) as (x, y, z) tuples, or None if no pose."""
    landmarks = pose_result.pose_landmarks
    if not landmarks:
        return None
    lm = landmarks[0]
    shoulder = (lm[_RIGHT_SHOULDER].x, lm[_RIGHT_SHOULDER].y, lm[_RIGHT_SHOULDER].z)
    elbow    = (lm[_RIGHT_ELBOW].x,    lm[_RIGHT_ELBOW].y,    lm[_RIGHT_ELBOW].z)
    wrist    = (lm[_RIGHT_WRIST].x,    lm[_RIGHT_WRIST].y,    lm[_RIGHT_WRIST].z)
    index    = (lm[_RIGHT_INDEX].x,    lm[_RIGHT_INDEX].y,    lm[_RIGHT_INDEX].z)
    return shoulder, elbow, wrist, index


def extract_full_pose_tuples(pose_result):
    """Return (opp_shoulder, shoulder, elbow, wrist, index) as (x, y, z) tuples, or None if no pose."""
    landmarks = pose_result.pose_landmarks
    if not landmarks:
        return None
    lm = landmarks[0]
    opp_shoulder = (lm[_LEFT_SHOULDER].x,  lm[_LEFT_SHOULDER].y,  lm[_LEFT_SHOULDER].z)
    shoulder     = (lm[_RIGHT_SHOULDER].x, lm[_RIGHT_SHOULDER].y, lm[_RIGHT_SHOULDER].z)
    elbow        = (lm[_RIGHT_ELBOW].x,    lm[_RIGHT_ELBOW].y,    lm[_RIGHT_ELBOW].z)
    wrist        = (lm[_RIGHT_WRIST].x,    lm[_RIGHT_WRIST].y,    lm[_RIGHT_WRIST].z)
    index        = (lm[_RIGHT_INDEX].x,    lm[_RIGHT_INDEX].y,    lm[_RIGHT_INDEX].z)
    return opp_shoulder, shoulder, elbow, wrist, index


def extract_hand_tuples(hand_result):
    """Return (wrist, index_mcp, pinky_mcp) as (x, y, z) tuples, or None if no hand."""
    landmarks = hand_result.hand_landmarks
    if not landmarks:
        return None
    lm = landmarks[0]
    wrist     = (lm[_HAND_WRIST].x,     lm[_HAND_WRIST].y,     lm[_HAND_WRIST].z)
    index_mcp = (lm[_HAND_INDEX_MCP].x,  lm[_HAND_INDEX_MCP].y,  lm[_HAND_INDEX_MCP].z)
    pinky_mcp = (lm[_HAND_PINKY_MCP].x,  lm[_HAND_PINKY_MCP].y,  lm[_HAND_PINKY_MCP].z)
    return wrist, index_mcp, pinky_mcp
