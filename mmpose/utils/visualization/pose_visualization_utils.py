
import open3d
import numpy as np
from scipy.spatial.transform import Rotation

def get_sphere(position, radius=1.0, color=(0.1, 0.1, 0.7)):
    mesh_sphere: open3d.geometry.TriangleMesh = open3d.geometry.TriangleMesh.create_sphere(radius=radius,
                                                                                           resolution=5)
    mesh_sphere.paint_uniform_color(color)

    # translate to position
    mesh_sphere = mesh_sphere.translate(position, relative=False)
    return mesh_sphere

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def get_cylinder(start_point, end_point, radius=0.3, color=(0.1, 0.9, 0.1)):
    center = (start_point + end_point) / 2
    height = np.linalg.norm(start_point - end_point)
    mesh_cylinder: open3d.geometry.TriangleMesh = open3d.geometry.TriangleMesh.create_cylinder(radius=radius,
                                                                                               height=height,
                                                                                               resolution=5)
    mesh_cylinder.paint_uniform_color(color)

    # translate and rotate to position
    # rotate vector
    rot_vec = end_point - start_point
    rot_vec = rot_vec / np.linalg.norm(rot_vec)
    rot_0 = np.array([0, 0, 1])
    rot_mat = rotation_matrix_from_vectors(rot_0, rot_vec)
    # if open3d.__version__ >= '0.9.0.0':
    #     rotation_param = rot_mat
    # else:
    #     rotation_param = Rotation.from_matrix(rot_mat).as_euler('xyz')
    rotation_param = rot_mat
    mesh_cylinder = mesh_cylinder.rotate(rotation_param)
    mesh_cylinder = mesh_cylinder.translate(center, relative=False)
    return mesh_cylinder

if __name__ == '__main__':
    point1 = np.array([-1, 11, 8])
    point2 = np.array([12, -1, 5])
    sphere1 = get_sphere(position=point1, radius=0.1)
    sphere2 = get_sphere(position=point2, radius=0.1)
    cylinder = get_cylinder(start_point=point1, end_point=point2, radius=0.02)

    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    open3d.visualization.draw_geometries(
        [sphere1, sphere2, cylinder, mesh_frame])