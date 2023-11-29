import numpy as np
from scipy.spatial.transform import Rotation


# code modified from zaw lin
def get_cv_rt_from_blender(location, rotation):
    # bcam stands for blender camera
    R_bcam2cv = np.array(
        [[1, 0, 0],
         [0, -1, 0],
         [0, 0, -1]])

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints

    R_world2bcam = Rotation.from_euler('xyz', rotation, degrees=False).as_matrix().T

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam.dot(location)

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv.dot(R_world2bcam)
    T_world2cv = R_bcam2cv.dot(T_world2bcam)

    #put into 3x4 matrix
    mat = np.array([
        np.concatenate([R_world2cv[0], [T_world2cv[0]]]),
        np.concatenate([R_world2cv[1], [T_world2cv[1]]]),
        np.concatenate([R_world2cv[2], [T_world2cv[2]]]),
        [0, 0, 0, 1]
    ])
    return T_world2cv, R_world2cv, mat
