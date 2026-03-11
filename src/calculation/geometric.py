from src.calculation.vectors import *

_NOSE = 0
_LEFT_EYE_INNER = 1
_LEFT_EYE = 2
_LEFT_EYE_OUTER = 3
_RIGHT_EYE_INNER = 4
_RIGHT_EYE = 5
_RIGHT_EYE_OUTER = 6
_LEFT_EAR = 7
_RIGHT_EAR = 8
_MOUTH_LEFT = 9
_MOUTH_RIGHT = 10
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_PINKY = 17
_RIGHT_PINKY = 18
_LEFT_INDEX = 19
_RIGHT_INDEX = 20
_LEFT_THUMB = 21
_RIGHT_THUMB = 22
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_KNEE = 25
_RIGHT_KNEE = 26
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28
_LEFT_HEEL = 29
_RIGHT_HEEL = 30
_LEFT_FOOT = 31
_RIGHT_FOOT = 32

def _lm2pt(lm) -> Point_3D:
    return(lm.x, lm.y, lm.z)

def _center_vector(landmarks) -> Point_3D:
    hips = _hip_mid(landmarks)
    shoulders = _shoulder_mid(landmarks)
    
    return vector(hips, shoulders)

def _shoulder_mid(landmarks) -> Point_3D:
    left = landmarks[_LEFT_SHOULDER]
    right = landmarks[_RIGHT_SHOULDER]
    x = (left.x + right.x) / 2
    y = (left.y + right.y) / 2
    z = (left.z + right.z) / 2
    return (x, y, z)

def _shoulder_vector(landmarks) ->  Point_3D:
    right = landmarks[_RIGHT_SHOULDER]
    left = landmarks[_LEFT_SHOULDER]
    right_point = _lm2pt(right)
    left_point = _lm2pt(left)
    return vector(left_point, right_point)

def _hip_mid(landmarks) -> Point_3D:
    left = landmarks[_LEFT_HIP]
    right = landmarks[_RIGHT_HIP]
    x = (left.x + right.x) / 2
    y = (left.y + right.y) / 2
    z = (left.z + right.z) / 2
    return (x, y, z)

def _hip_vector(landmarks) -> Point_3D:
    right = landmarks[_RIGHT_HIP]
    left = landmarks[_LEFT_HIP]
    right_point = _lm2pt(right)
    left_point = _lm2pt(left)
    return vector(left_point, right_point)

def _arm_vector(landmarks) -> Point_3D:
    src = landmarks[_RIGHT_SHOULDER]
    end = landmarks[_RIGHT_ELBOW]
    src_point = _lm2pt(src)
    end_point = _lm2pt(end)
    return vector(src_point, end_point)

def _forearm_vector(landmarks) -> Point_3D:
    src = landmarks[_RIGHT_ELBOW]
    end = landmarks[_RIGHT_WRIST]
    src_point = _lm2pt(src)
    end_point = _lm2pt(end)
    return vector(src_point, end_point)

def _mid_mcp(landmarks) -> Point_3D:
    pinky = landmarks[_RIGHT_PINKY]
    index = landmarks[_RIGHT_INDEX]
    x = (pinky.x + index.x) / 2
    y = (pinky.y + index.y) / 2
    z = (pinky.z + index.z) / 2
    return (x, y, z)

def _wrist_vector(landmarks) -> Point_3D:
    src = landmarks[_RIGHT_WRIST]
    end = _mid_mcp(landmarks)
    src_point = _lm2pt(src)
    return vector(src_point, end)

def _mcp_vector(landmarks) -> Point_3D:
    pinky = landmarks[_RIGHT_PINKY]
    index = landmarks[_RIGHT_INDEX]
    pinky_pt = _lm2pt(pinky)
    index_pt = _lm2pt(index)
    return vector(pinky_pt, index_pt)

class human_angles:
    def __init__(self):
        '''
        link vectors
        '''
        self.center: Point_3D
        self.shoulder: Point_3D
        self.arm: Point_3D
        self.elbow: Point_3D
        self.wrist: Point_3D
        self.mcp: Point_3D

    def update_vectors(self, landmarks):
        self.center = _center_vector(landmarks)
        self.shoulder = _shoulder_vector(landmarks)
        self.arm = _arm_vector(landmarks)
        self.forearm = _forearm_vector(landmarks)
        self.wrist = _wrist_vector(landmarks)
        self.mcp = _mcp_vector(landmarks)
        
    def theta_1(self) -> float:
        '''
        shoulder rot
        '''
        _, y1, z1 = self.center
        _, y2, z2 = self.arm
        return vec_angle_2d((y1, z1), (y2, z2))

    def theta_2(self) -> float:
        '''
        shoulder flex
        '''
        return vec_angle_3d(self.shoulder, self.arm)

    def theta_3(self) -> float:
        '''
        arm rot
        '''
        return vec_plane_angle_3d(self.forearm, self.arm, self.shoulder)

    def theta_4(self) -> float:
        '''
        elbow flex
        '''
        x1, y1, _ = self.arm
        x2, y2, _ = self.forearm
        return vec_angle_2d((x1, y1), (x2, y2))

    def theta_5(self) -> float:
        '''
        forearm rot
        '''
        return vec_plane_angle_3d(self.mcp, self.arm, self.forearm)

    def theta_6(self) -> float:
        '''
        wrist flex
        '''
        return vec_angle_3d(self.forearm, self.wrist)

    def theta_7(self) -> float:
        '''
        wrist flex
        '''
        return vec_angle_3d(self.forearm, self.wrist)