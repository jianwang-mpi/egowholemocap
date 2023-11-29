#  Copyright Jian Wang @ MPI-INF (c) 2023.
import os
import pickle

import numpy as np
from tqdm import tqdm

from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import (CalculateEgocentricCameraLocationFromSMPLX, Collect,
                                       ConvertSMPLXOutputToEgocentricCameraLocation,
                                       SplitEgoHandMotion, SplitGlobalSMPLXJoints,
                                       PreProcessHandMotion,
                                       PreProcessMo2Cap2BodyMotion)


def calculate_egobody_mean_std():
    seq_len = 196
    split_sequence = False
    normalize = False
    egobody_pipeline = [
        CalculateEgocentricCameraLocationFromSMPLX(random_camera_rotation=True, random_camera_translation=True),
        ConvertSMPLXOutputToEgocentricCameraLocation(),
        SplitGlobalSMPLXJoints(smplx_joint_name='ego_smplx_joints'),
        PreProcessHandMotion(normalize=normalize),
        PreProcessMo2Cap2BodyMotion(normalize=normalize),
        Collect(keys=['smplx_output', 'smplx_input', 'ego_camera_rot', 'ego_camera_transl', 'ego_camera_transform',
                      'ego_smplx_joints', 'ego_smplx_vertices',
                      'left_hand_keypoints_3d', 'right_hand_keypoints_3d',
                      'left_hand_features', 'right_hand_features', 'mo2cap2_body_features'],
                meta_keys=['smplx_input', 'ego_camera_rot', 'ego_camera_transl', 'ego_camera_transform',
                           'ego_smplx_joints',])
    ]

    dataset_cfg = dict(
        type='EgoBodyDataset',
        data_path=data_path,
        seq_len=seq_len,
        smplx_model_dir=smplx_model_dir,
        pipeline=egobody_pipeline,
        split_sequence=split_sequence,
        skip_frames=seq_len,
        data_dirs=None,  # none means use all data
        test_mode=True
    )

    egobody_dataset = build_dataset(dataset_cfg)

    print(f'length of dataset is: {len(egobody_dataset)}')

    left_hand_features_all = []
    right_hand_features_all = []
    mo2cap2_body_features_all = []

    left_hand_features_seq_all = []
    right_hand_features_seq_all = []
    mo2cap2_body_features_seq_all = []
    ego_smplx_joints_seq_all = []
    image_metas_seq_all = []

    for seq_id in tqdm(range(len(egobody_dataset))):
        data_i = egobody_dataset[seq_id]
        left_hand_features = data_i['left_hand_features']
        right_hand_features = data_i['right_hand_features']
        mo2cap2_body_features = data_i['mo2cap2_body_features']

        left_hand_features_all.extend(left_hand_features.numpy())
        right_hand_features_all.extend(right_hand_features.numpy())
        mo2cap2_body_features_all.extend(mo2cap2_body_features.numpy())

        ego_smplx_joints = data_i['ego_smplx_joints']
        img_metas = data_i['img_metas'].data
        ego_smplx_joints_seq_all.append(ego_smplx_joints.numpy())
        left_hand_features_seq_all.append(left_hand_features.numpy())
        right_hand_features_seq_all.append(right_hand_features.numpy())
        mo2cap2_body_features_seq_all.append(mo2cap2_body_features.numpy())
        image_metas_seq_all.append(img_metas)


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

    features_seqs = {
        'left_hand_features_seqs': left_hand_features_seq_all,
        'right_hand_features_seqs': right_hand_features_seq_all,
        'mo2cap2_body_features_seqs': mo2cap2_body_features_seq_all,
        'ego_smplx_joints_seqs': ego_smplx_joints_seq_all,
        'image_metas_seqs': image_metas_seq_all,
    }
    with open(save_mean_std_path, 'wb') as f:
        pickle.dump(mean_and_std, f)

    with open(save_data_path, 'wb') as f:
        pickle.dump(features_seqs, f)


if __name__ == '__main__':
    if os.name == 'nt':
        data_path = r'Z:\datasets04\static00\EgoBody'
        smplx_model_dir = r'Z:\EgoMocap\work\EgocentricFullBody\human_models\smplx_new'
        save_mean_std_path = r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\egobody\ego_mean_std.pkl'
        save_data_path = r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\egobody\ego_smplx_joints.pkl'
    else:
        data_path = '/CT/datasets04/static00/EgoBody'
        smplx_model_dir = '/CT/EgoMocap/work/EgocentricFullBody/human_models/smplx_new'
        save_mean_std_path = '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/egobody/ego_mean_std.pkl'
        save_data_path = '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/egobody/ego_smplx_joints.pkl'

    calculate_egobody_mean_std()
