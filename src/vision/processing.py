import numpy as np

'''
x and y: Landmark coordinates normalized between 0.0 and 1.0 by the image width (x) and height (y).

z: The landmark depth, with the depth at the midpoint of the hips as the origin. The smaller the value, 
the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
'''

def generate_elbow_angle(shoulder: tuple[float, float, float],
                         elbow: tuple[float, float, float],
                         wrist: tuple[float, float, float]) -> float:
    '''
    create vectors: elbow -> shoulder, elbow -> wrist
    cross product magnitude
    '''
    e = np.array(elbow)
    v1 = np.array(shoulder) - e
    v2 = np.array(wrist) - e

    cross = np.cross(v1, v2)
    cross_mag = np.linalg.norm(cross)
    dot = np.dot(v1, v2)

    return float(np.arctan2(cross_mag, dot))