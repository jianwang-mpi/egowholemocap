import numpy as np
import cv2
import open3d

from mmpose.utils.visualization.pose_visualization_utils import get_sphere, get_cylinder

lines = [(0, 1, 'right'), (0, 4, 'left'), (1, 2, 'right'), (2, 3, 'right'), (4, 5, 'left'), (5, 6, 'left'),
         (1, 7, 'right'), (4, 11, 'left'), (7, 8, 'right'), (8, 9, 'right'), (9, 10, 'right'),
         (11, 12, 'left'), (12, 13, 'left'), (13, 14, 'left'), (7, 11, 'left')]


def draw_joints(joints, img, color=(0, 255, 0), right_color=(255, 0, 0)):
    joints_num = joints.shape[0]
    for line in lines:
        if line[0] < joints_num and line[1] < joints_num:
            start = joints[line[0]].astype(np.int32)
            end = joints[line[1]].astype(np.int32)
            left_or_right = line[2]
            if left_or_right == 'right':
                paint_color = right_color
            else:
                paint_color = color
            img = cv2.line(img, (start[0], start[1]), (end[0], end[1]), color=paint_color, thickness=4)
    for j in range(joints_num):
        img = cv2.circle(img, center=(joints[j][0].astype(np.int32), joints[j][1].astype(np.int32)),
                         radius=2, color=(0, 0, 255), thickness=6)

    return img


def draw_keypoints(joints, img, radius=2, color=(0, 0, 255)):
    joints_num = joints.shape[0]
    for j in range(joints_num):
        img = cv2.circle(img, center=(joints[j][0].astype(np.int32), joints[j][1].astype(np.int32)),
                         radius=radius, color=color, thickness=radius + 1)

    return img


def draw_bbox(img, up_left, bottom_right):
    up_left = up_left.astype(np.int32)
    bottom_right = bottom_right.astype(np.int32)
    img = cv2.rectangle(img, up_left, bottom_right, color=(255, 0, 0), thickness=3)
    return img


def draw_keypoints_3d(skeleton, joint_color=(0.1, 0.1, 0.7), radius=0.03):
    final_mesh = open3d.geometry.TriangleMesh()
    for i in range(len(skeleton)):
        keypoint_mesh = get_sphere(position=skeleton[i], radius=radius, color=joint_color)
        final_mesh = final_mesh + keypoint_mesh

    return final_mesh


def draw_skeleton_with_chain(skeleton, chain_list, joint_color=(0.1, 0.1, 0.7), bone_color=(0.1, 0.9, 0.1),
                             keypoint_radius=0.03, line_radius=0.0075):
    final_mesh = open3d.geometry.TriangleMesh()
    for i in range(len(skeleton)):
        keypoint_mesh = get_sphere(position=skeleton[i], radius=keypoint_radius, color=joint_color)
        final_mesh = final_mesh + keypoint_mesh

    for chain in chain_list:
        for i in range(len(chain) - 1):
            start_point = skeleton[chain[i]]
            end_point = skeleton[chain[i + 1]]
            line_mesh = get_cylinder(start_point, end_point, radius=line_radius, color=bone_color)
            final_mesh += line_mesh

    return final_mesh


def draw_geometries(pcds):
    """
    Draw Geometries
    Args:
        - pcds (): [pcd1,pcd2,...]
    """
    open3d.visualization.draw_geometries(pcds)


def get_o3d_FOR(origin=[0, 0, 0], size=10):
    """
    Create a FOR that can be added to the open3d point cloud
    """
    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size)
    mesh_frame.translate(origin)
    return (mesh_frame)


def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec ():
    """
    magnitude = np.sqrt(np.sum(vec ** 2))
    return (magnitude)


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the
    z axis vector of the original FOR. The first rotation that is
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis.

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec ():
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1] / vec[0])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T @ vec.reshape(-1, 1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0] / vec[2])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    return (Rz, Ry)


def create_arrow(scale=10):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale * 0.2
    cylinder_height = scale * 0.8
    cone_radius = scale / 10
    cylinder_radius = scale / 20
    mesh_frame = open3d.geometry.TriangleMesh.create_arrow(cone_radius=1,
                                                           cone_height=cone_height,
                                                           cylinder_radius=0.5,
                                                           cylinder_height=cylinder_height)
    return (mesh_frame)


def get_arrow(origin=[0, 0, 0], end=None, vec=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 10
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return (mesh)
