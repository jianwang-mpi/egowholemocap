#  Copyright Jian Wang @ MPI-INF (c) 2023.
import os
import pickle
from copy import deepcopy

import cv2
import numpy as np
import open3d
import smplx
import torch

from mmpose.data.keypoints_mapping.mano import mano_skeleton
from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import (CalculateEgocentricCameraLocationFromSMPLX, Collect,
                                       ConvertSMPLXOutputToEgocentricCameraLocation,
                                       SplitEgoHandMotion, PreProcessRootMotion,
                                       PreProcessHandMotion, PreProcessMo2Cap2BodyMotion, EgoFeaturesNormalize)
from mmpose.models.diffusion_mdm.data_loaders.humanml.common.quaternion import euler_to_quaternion, qmul_np, qrot_np, \
    qinv_np
from mmpose.utils.visualization.draw import draw_keypoints_3d, draw_skeleton_with_chain
from mmpose.utils.visualization.skeleton import Skeleton

if os.name == 'nt':
    data_root = r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\diffusion_fullbody\features_seqs_196.pkl'
    mean_std_path = r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\diffusion_fullbody\mean_std.pkl'
else:
    data_root = '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/diffusion_fullbody/features_seqs_196.pkl'
    mean_std_path = '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/diffusion_fullbody/mean_std.pkl'

def get_global_root_rot_trans(local_root_velocity, local_root_rot_velocity):
    init_root_quat = np.array([1, 0, 0, 0])
    init_root_trans = np.array([0, 0, 0])
    # recover the root rotation for each frame
    seq_len = len(local_root_rot_velocity)
    root_quat = np.zeros((seq_len + 1, 4))
    root_trans = np.zeros((seq_len + 1, 3))
    root_quat[0] = init_root_quat
    root_trans[0] = init_root_trans
    for i in range(seq_len):
        root_quat[i + 1] = qmul_np(local_root_rot_velocity[i], root_quat[i])
        root_trans[i + 1] = qrot_np(qinv_np(root_quat[i]), local_root_velocity[i]) + root_trans[i]
    return -root_trans, root_quat

def try_egobody_dataset(seq_id):
    seq_len = 196

    pipeline = [
        EgoFeaturesNormalize(mean_std_path=mean_std_path,

                             normalize_name_list=['left_hand_features',
                                                  'right_hand_features',
                                                  'mo2cap2_body_features',
                                                  'root_features']),
        Collect(keys=['left_hand_features',
                      'right_hand_features',
                      'mo2cap2_body_features',
                      'root_features'],
                meta_keys=['global_smplx_joints'])
    ]

    dataset_cfg = dict(
        type='EgoSMPLXFeaturesDataset',
        data_path_list=data_root,
        seq_len=seq_len,
        pipeline=pipeline,
        split_sequence=True,
        skip_frame=5,
        test_mode=True
    )

    dataset = build_dataset(dataset_cfg)

    print(f'length of dataset is: {len(dataset)}')

    data_i = dataset[seq_id]

    left_hand_features = data_i['left_hand_features']
    right_hand_features = data_i['right_hand_features']
    mo2cap2_body_features = data_i['mo2cap2_body_features']
    root_features = data_i['root_features']

    with open(mean_std_path, 'rb') as f:
        mean_std = pickle.load(f)
    left_hand_mean = mean_std['left_hand_mean']
    left_hand_std = mean_std['left_hand_std']
    right_hand_mean = mean_std['right_hand_mean']
    right_hand_std = mean_std['right_hand_std']
    mo2cap2_body_mean = mean_std['mo2cap2_body_mean']
    mo2cap2_body_std = mean_std['mo2cap2_body_std']
    root_features_mean = mean_std['root_features_mean']
    root_features_std = mean_std['root_features_std']

    left_hand_keypoints_3d = left_hand_features * left_hand_std + left_hand_mean
    left_hand_keypoints_3d = np.reshape(left_hand_keypoints_3d, (-1, 21, 3))
    right_hand_keypoints_3d = right_hand_features * right_hand_std + right_hand_mean
    right_hand_keypoints_3d = np.reshape(right_hand_keypoints_3d, (-1, 21, 3))
    left_hand_keypoints_3d[:, 1:] += deepcopy(left_hand_keypoints_3d[:, 0:1])
    right_hand_keypoints_3d[:, 1:] += deepcopy(right_hand_keypoints_3d[:, 0:1])
    mo2cap2_body_keypoints_3d = mo2cap2_body_features * mo2cap2_body_std + mo2cap2_body_mean
    mo2cap2_body_keypoints_3d = np.reshape(mo2cap2_body_keypoints_3d, (-1, 15, 3))
    combined_motion = np.concatenate([mo2cap2_body_keypoints_3d, left_hand_keypoints_3d, right_hand_keypoints_3d],
                                     axis=1)

    root_features = root_features * root_features_std + root_features_mean
    root_velocity_xz = root_features[:, 0:2]
    root_velocity_xyz = np.zeros((root_velocity_xz.shape[0], 3))
    root_velocity_xyz[:, 0] = root_velocity_xz[:, 0]
    root_velocity_xyz[:, 2] = root_velocity_xz[:, 1]
    root_rotation_velocity_y = root_features[:, 2:3]
    euler_root_rotation_velocity = np.zeros((root_rotation_velocity_y.shape[0], 3))
    euler_root_rotation_velocity[:, 1] = 1
    euler_root_rotation_velocity = euler_root_rotation_velocity * root_rotation_velocity_y
    quat_root_rotation_velocity = euler_to_quaternion(euler_root_rotation_velocity, order='xyz')

    root_trans, root_rot = get_global_root_rot_trans(root_velocity_xyz, quat_root_rotation_velocity)

    root_trans = root_trans[:-1]
    root_rot = root_rot[:-1]
    rot_quat_inv_list = qinv_np(root_rot)
    rot_quat_inv_list = np.repeat(rot_quat_inv_list[:, None, :], combined_motion.shape[1], axis=1)
    combined_motion = qrot_np(rot_quat_inv_list, combined_motion)
    if len(root_trans.shape) == 2:
        root_trans = root_trans[:, None, :]
    combined_motion = combined_motion - root_trans

    for frame_id in [0, 10, 20, 30, 40, 50]:
        mo2cap2_body_keypoints_3d = combined_motion[frame_id, :15]
        left_hand_keypoints_3d = combined_motion[frame_id, 15:15 + 21]
        right_hand_keypoints_3d = combined_motion[frame_id, 15 + 21:15 + 21 * 2]

        left_hand_mesh = draw_skeleton_with_chain(left_hand_keypoints_3d, mano_skeleton, keypoint_radius=0.01,
                                                  line_radius=0.0025)
        right_hand_mesh = draw_skeleton_with_chain(right_hand_keypoints_3d, mano_skeleton, keypoint_radius=0.01,
                                                  line_radius=0.0025)
        mo2cap2_body_mesh = draw_skeleton_with_chain(mo2cap2_body_keypoints_3d, mo2cap2_chain)

        # visualize body
        coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        open3d.visualization.draw_geometries([left_hand_mesh, right_hand_mesh, mo2cap2_body_mesh, coord])


if __name__ == '__main__':
    try_egobody_dataset(91000)

