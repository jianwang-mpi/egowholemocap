#  Copyright Jian Wang @ MPI-INF (c) 2023.
import os
import pickle

import numpy as np
from tqdm import tqdm

from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import (CalculateEgocentricCameraLocationFromSMPLX, Collect,
                                       ConvertSMPLXOutputToEgocentricCameraLocation,
                                       SplitEgoHandMotion,
                                       PreProcessHandMotion,
                                       PreProcessMo2Cap2BodyMotion, AlignGlobalSMPLXJoints, SplitGlobalSMPLXJoints)


def calculate_egobody_mean_std():
    seq_len = 196
    pipeline = [
        AlignGlobalSMPLXJoints(align_every_joint=True),
        SplitGlobalSMPLXJoints(),
        PreProcessHandMotion(normalize=False),
        PreProcessMo2Cap2BodyMotion(normalize=False),
        Collect(keys=['smplx_output', 'smplx_input',
                      'global_smplx_joints', 'global_smplx_vertices', 'aligned_smplx_joints',
                      'mo2cap2_body_features', 'left_hand_features', 'right_hand_features', 'json_path_calib',
                      'processed_left_hand_keypoints_3d', 'processed_right_hand_keypoints_3d',],
                meta_keys=[])
    ]

    dataset_cfg = dict(
        type='EgoBodyDataset',
        data_path=data_path,
        seq_len=seq_len,
        smplx_model_dir=smplx_model_dir,
        pipeline=pipeline,
        split_sequence=False,
        skip_frames=seq_len,  # do not use skip frames
        data_dirs=None,  # none means use all data
        test_mode=True)

    egobody_dataset = build_dataset(dataset_cfg)

    print(f'length of dataset is: {len(egobody_dataset)}')

    left_hand_features_all = []
    right_hand_features_all = []
    mo2cap2_body_features_all = []
    aligned_smplx_joints_all = []

    for seq_id in tqdm(range(len(egobody_dataset))):
        data_i = egobody_dataset[seq_id]
        left_hand_features = data_i['left_hand_features']
        right_hand_features = data_i['right_hand_features']
        mo2cap2_body_features = data_i['mo2cap2_body_features']
        left_hand_features_all.extend(left_hand_features.numpy())
        right_hand_features_all.extend(right_hand_features.numpy())
        mo2cap2_body_features_all.extend(mo2cap2_body_features.numpy())

        aligned_smplx_joints = data_i['aligned_smplx_joints']
        aligned_smplx_joints_all.append(aligned_smplx_joints.numpy())


    left_hand_features_all = np.asarray(left_hand_features_all)
    right_hand_features_all = np.asarray(right_hand_features_all)
    mo2cap2_body_features_all = np.asarray(mo2cap2_body_features_all)

    left_hand_features_mean = np.mean(left_hand_features_all, axis=0)
    left_hand_features_std = np.std(left_hand_features_all, axis=0)
    right_hand_features_mean = np.mean(right_hand_features_all, axis=0)
    right_hand_features_std = np.std(right_hand_features_all, axis=0)

    mo2cap2_body_features_mean = np.mean(mo2cap2_body_features_all, axis=0)
    mo2cap2_body_features_std = np.std(mo2cap2_body_features_all, axis=0)

    print(f'left_hand_features_mean: {left_hand_features_mean}')
    print(f'left_hand_features_std: {left_hand_features_std}')
    print(f'right_hand_features_mean: {right_hand_features_mean}')
    print(f'right_hand_features_std: {right_hand_features_std}')
    print(f'mo2cap2_body_features_mean: {mo2cap2_body_features_mean}')
    print(f'mo2cap2_body_features_std: {mo2cap2_body_features_std}')

    mean_and_std = {
        'left_hand_mean': left_hand_features_mean,
        'left_hand_std': left_hand_features_std,
        'right_hand_mean': right_hand_features_mean,
        'right_hand_std': right_hand_features_std,
        'mo2cap2_body_mean': mo2cap2_body_features_mean,
        'mo2cap2_body_std': mo2cap2_body_features_std
    }


    with open(save_mean_std_path, 'wb') as f:
        pickle.dump(mean_and_std, f)

    with open(save_data_path, 'wb') as f:
        pickle.dump(aligned_smplx_joints, f)



if __name__ == '__main__':
    if os.name == 'nt':
        data_path = r'Z:\datasets04\static00\EgoBody'
        smplx_model_dir = r'Z:\EgoMocap\work\EgocentricFullBody\human_models\smplx_new'
        save_mean_std_path = r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\egobody\global_aligned_mean_std.pkl'
        save_data_path = r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\egobody\global_aligned_smplx_joints.pkl'
    else:
        data_path = '/CT/datasets04/static00/EgoBody'
        smplx_model_dir = '/CT/EgoMocap/work/EgocentricFullBody/human_models/smplx_new'
        save_mean_std_path = '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/egobody/global_aligned_mean_std.pkl'
        save_data_path = '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/egobody/global_aligned_smplx_joints.pkl'

    calculate_egobody_mean_std()
