import math
import numpy as np

Point_2D = tuple[float, float]
Point_3D = tuple[float, float, float]

def vector(a: Point_3D,
           b: Point_3D
           ) -> Point_3D:
    '''
    Get vector from a to b
    '''
    a_x, a_y, a_z = a
    b_x, b_y, b_z = b
    return (b_x - a_x, b_y - a_y, b_z - a_z)

def norm_vector(a: Point_3D,
                    b: Point_3D
                    ) -> Point_3D:
    '''
    Get normalized vector from a to b
    '''
    vec = np.array(vector(a, b))
    norm = np.linalg.norm(vec)
    if norm == 0:
        return tuple(vec)
    return tuple(vec/norm)

def vector_2d(a: Point_2D,
              b: Point_2D
              ) -> Point_2D:
    '''
    Get 2d vector from a to b
    '''
    a_x, a_y = a
    b_x, b_y = b
    return (b_x - a_x, b_y - a_y)

def norm_vector_2d(a: Point_2D,
                   b: Point_2D
                   ) -> float:
    '''
    Get norm of 2d vector
    '''
    a_x, a_y = a
    b_x, b_y = b
    return math.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)

def vec_angle_2d(v1: Point_2D,
                 v2: Point_2D
                 ) -> float:
    '''
    Calculate angle between two 2D vectors, range (0, 2pi) 
    '''
    x1, y1 = v1
    x2, y2 = v2
    return (math.atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2) + 2 * math.pi) % (2 * math.pi) 

def vec_angle_2d_centered(v1: Point_2D,
                          v2: Point_2D
                          ) -> float:
    '''
    Calculate angle between two 2D vectors, range (-pi, pi) 
    '''
    x1, y1 = v1
    x2, y2 = v2
    return math.atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2) 

def vec_angle_3d(v1: Point_3D,
                 v2: Point_3D
                 ) -> float:
    v1_np = np.array(v1)
    v2_np = np.array(v2)
    crossmag = np.linalg.norm(np.cross(v1_np, v2_np))
    dot = np.dot(v1_np, v2_np)
    return np.atan2(crossmag, dot)

def vec_plane_angle_3d(v: Point_3D,
                       a: Point_3D,
                       b: Point_3D) -> float:
    a_np = np.array(a)
    b_np = np.array(b)
    v_np = np.array(v)
    n = np.cross(a, b)
    dot = np.dot(v_np, n)
    crossmag = np.linalg.norm(np.cross(v_np, n))
    return math.atan2(dot, crossmag)

if __name__ == '__main__':
    x = (3, 2, 1)
    y = (1, 2, 3)
    assert (-2, 0, 2) == vector(x, y)
    assert (-1/math.sqrt(2), 0, 1/math.sqrt(2)) == norm_vector(x, y)
    assert 0 == vec_angle_2d((1,1), (1,1))
    assert math.pi/2 == vec_angle_2d((1,0), (0,1))
    assert math.pi == vec_angle_2d((-1,0), (1,0))
    assert math.pi * 1.5 == vec_angle_2d((0,-1), (-1, 0))
    assert math.pi / 2 == vec_angle_3d((1, 0, 0), (0, 1, 0))

    print(vec_angle_2d((0, 1), (1, 0)))