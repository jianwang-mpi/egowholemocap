#  Copyright Jian Wang @ MPI-INF (c) 2023.
import pickle

import cv2
import numpy as np
import open3d
import smplx
import torch

from mmpose.data.keypoints_mapping.mano import mano_skeleton
from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import (CalculateEgocentricCameraLocationFromSMPLX, Collect,
                                       ConvertSMPLXOutputToEgocentricCameraLocation,
                                       SplitEgoHandMotion,
                                       PreProcessHandMotion, PreProcessMo2Cap2BodyMotion)
from mmpose.utils.visualization.draw import draw_keypoints_3d, draw_skeleton_with_chain
from mmpose.utils.visualization.skeleton import Skeleton

def try_egobody_dataset(seq_id):
    seq_len = 196
    mean_std_path = r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\egobody\mean_std.pkl'
    pipeline = [
        SplitEgoHandMotion(),
        PreProcessHandMotion(normalize=True, mean_std_path=mean_std_path),
        PreProcessMo2Cap2BodyMotion(normalize=True, mean_std_path=mean_std_path),
        Collect(keys=['ego_smplx_joints', 'left_hand_features',
                      'right_hand_features', 'mo2cap2_body_features'],
                meta_keys=[])
    ]

    dataset_cfg = dict(
        type='EgoSMPLXDataset',
        data_path_list=[{
            'path': r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\egobody\smplx_joints.pkl',
            'frame_rate': 30
        }],
        seq_len=seq_len,
        pipeline=pipeline,
        split_sequence=True,
        target_frame_rate=25,
        skip_frame=5,
        test_mode=True
    )

    egobody_dataset = build_dataset(dataset_cfg)

    print(f'length of dataset is: {len(egobody_dataset)}')

    data_i = egobody_dataset[seq_id]


    ego_smplx_joints = data_i['ego_smplx_joints']

    left_hand_features = data_i['left_hand_features']
    right_hand_features = data_i['right_hand_features']

    with open(mean_std_path, 'rb') as f:
        mean_std = pickle.load(f)
    left_hand_mean = mean_std['left_hand_mean']
    left_hand_std = mean_std['left_hand_std']
    right_hand_mean = mean_std['right_hand_mean']
    right_hand_std = mean_std['right_hand_std']
    mo2cap2_body_mean = mean_std['mo2cap2_body_mean']
    mo2cap2_body_std = mean_std['mo2cap2_body_std']

    for frame_id in [0, 50, 100, 150]:
        # visualize hands
        left_hand_keypoints_3d = left_hand_features[frame_id].numpy() * left_hand_std + left_hand_mean
        right_hand_keypoints_3d = right_hand_features[frame_id].numpy() * right_hand_std + right_hand_mean

        print(left_hand_keypoints_3d)

        left_hand_keypoints_3d = left_hand_keypoints_3d.reshape(21, 3)
        right_hand_keypoints_3d = right_hand_keypoints_3d.reshape(21, 3)

        # left_hand_keypoints_3d[0] = np.array([0, 0, 0])
        # right_hand_keypoints_3d[0] = np.array([0, 0, 0])

        left_hand_keypoints_3d[1:] += left_hand_keypoints_3d[0:1]
        right_hand_keypoints_3d[1:] += right_hand_keypoints_3d[0:1]

        left_hand_mesh = draw_skeleton_with_chain(left_hand_keypoints_3d, mano_skeleton, keypoint_radius=0.01,
                                                  line_radius=0.0025)
        right_hand_mesh = draw_skeleton_with_chain(right_hand_keypoints_3d, mano_skeleton, keypoint_radius=0.01,
                                                  line_radius=0.0025)


        # visualize body
        open3d.visualization.draw_geometries([left_hand_mesh, right_hand_mesh])




if __name__ == '__main__':
    try_egobody_dataset(0)

