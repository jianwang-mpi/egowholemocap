#  Copyright Jian Wang @ MPI-INF (c) 2023.

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import mmpose.utils.geometry_utils.geometry_utils as gu
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated


def cartesian_to_spherical(points3d):
    # points3d shape: (n, 3)
    distance = np.linalg.norm(points3d, axis=-1)
    distance_2d = np.linalg.norm(points3d[:, :2], axis=-1)
    lat = np.arcsin(
        points3d[:, 2] / distance
    )
    lon = np.arcsin(
        points3d[:, 1] / distance_2d
    )
    return lat, lon


def spherical_to_rotation_matrix(lat, lon):
    # lat, lon shape: (n,)
    res = np.empty((lat.shape[0], 3, 3))
    for i in range(len(lat)):
        rot_i = R.from_euler('xyz', angles=[lat[i], 0, lon[i]], degrees=False)
        rot_mat_i = rot_i.as_matrix()
        res[i] = rot_mat_i
    return res


def mano_to_fisheye_camera_space(hand_pred, bbox_center, fisheye_camera_param):
    """
    transfer the hand prediction from the cropped weak perspective camera space to the fisheye camera space

    """
    fisheye_camera = FishEyeCameraCalibrated(calibration_file_path=fisheye_camera_param)
    depth = np.ones((bbox_center.shape[0],))
    point3d = fisheye_camera.camera2world(bbox_center, depth)
    local_to_fisheye_rotmat = spherical_to_rotation_matrix(*cartesian_to_spherical(point3d))

    hand_mano_rot, hand_mano_pose = hand_pred['mano_pose'][:, 3], hand_pred['mano_pose'][:, 3:]

    hand_mano_rotmat = gu.angle_axis_to_rotation_matrix(hand_mano_rot)

    local_to_fisheye_rotmat_torch = torch.from_numpy(local_to_fisheye_rotmat).float().to(hand_mano_rotmat.device)

    hand_global_rotmat = torch.bmm(hand_mano_rotmat, local_to_fisheye_rotmat_torch)

    hand_global_rot = gu.rotation_matrix_to_angle_axis(hand_global_rotmat)

    hand_mano_new_pose = torch.cat([hand_global_rot, hand_mano_pose], dim=-1)

    return hand_mano_new_pose

def hand_keypoints_3d_to_fisheye_camera_space(keypoints_3d, bbox_center, fisheye_camera_param):
    fisheye_camera = FishEyeCameraCalibrated(calibration_file_path=fisheye_camera_param)
    depth = np.ones((bbox_center.shape[0],))
    bbox_center_3d = fisheye_camera.camera2world(bbox_center, depth)

    local_to_fisheye_rotmat = spherical_to_rotation_matrix(*cartesian_to_spherical(bbox_center_3d))
    local_to_fisheye_rotmat = torch.from_numpy(local_to_fisheye_rotmat).float().to(keypoints_3d.device)
    # point3d_fisheye = torch.bmm(local_to_fisheye_rotmat, keypoints_3d.transpose(1, 2)).transpose(1, 2)
    point3d_fisheye = torch.bmm(keypoints_3d, local_to_fisheye_rotmat.transpose(1, 2))
    return point3d_fisheye
