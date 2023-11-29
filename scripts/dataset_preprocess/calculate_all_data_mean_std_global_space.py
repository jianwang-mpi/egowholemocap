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
                                       PreProcessMo2Cap2BodyMotion, AlignGlobalSMPLXJoints, SplitGlobalSMPLXJoints,
                                       AlignAllGlobalSMPLXJointsWithInfo, PreProcessRootMotion,
                                       AlignAllGlobalSMPLXJointsWithGlobalInfo)



def calculate_egobody_mean_std():
    seq_len = 196
    split_sequence=False
    normalize=False
    save_mean_std_path=os.path.join(save_dir, 'mean_std.pkl')
    save_data_path = os.path.join(save_dir, f'features_seqs_{seq_len}.pkl')
    pipeline = [
        AlignAllGlobalSMPLXJointsWithGlobalInfo(use_default_floor_height=True),
        SplitGlobalSMPLXJoints(smplx_joint_name='aligned_smplx_joints'),
        PreProcessHandMotion(normalize=normalize,
                             mean_std_path=save_mean_std_path),
        PreProcessMo2Cap2BodyMotion(normalize=normalize,
                                    mean_std_path=save_mean_std_path),
        PreProcessRootMotion(normalize=normalize,
                                mean_std_path=save_mean_std_path),
        Collect(keys=['aligned_smplx_joints',
                      'mo2cap2_body_features', 'left_hand_features', 'right_hand_features', 'root_features',
                      'processed_left_hand_keypoints_3d', 'processed_right_hand_keypoints_3d'],
                meta_keys=['root_trans_init_xz', 'root_rot_quat_init',
                           'root_trans_xz', 'root_rot_quat', 'local_root_velocity', 'local_joints_velocity',
                           'local_root_rotation_velocity', 'local_root_rotation_velocity_y',
                           'global_smplx_joints'])
    ]

    renderpeople_dataset_config = dict(
        type='RenderpeopleMotionDataset',
        data_path=renderpeople_data_path,
        seq_len=seq_len,
        skip_frames=seq_len,
        pipeline=pipeline,
        split_sequence=split_sequence,
        human_names=['render_people_adanna_joints_all'],
        test_mode=True
    )
    studio_motion_dataset_config = dict(
        type='MocapStudioMotionDataset',
        seq_len=seq_len,
        skip_frames=seq_len,
        pipeline=pipeline,
        split_sequence=split_sequence,
        test_mode=False,
        local=False,
        data_cfg={
        }
    )
    egobody_dataset_config = dict(
        type='EgoBodyDataset',
        data_path=egobody_data_path,
        seq_len=seq_len,
        smplx_model_dir=smplx_model_dir,
        pipeline=pipeline,
        split_sequence=split_sequence,
        skip_frames=seq_len,  # do not use skip frames
        data_dirs=None,  # none means use all data
        # data_dirs=['smplx_interactee_val'],
        place_on_floor=True, # place the human body on floor
        test_mode=True)

    dataset_cfg = dict(
        type='ConcatDataset',
        datasets=[egobody_dataset_config, renderpeople_dataset_config, studio_motion_dataset_config],
    )

    dataset = build_dataset(dataset_cfg)

    print(f'length of dataset is: {len(dataset)}')

    left_hand_features_all = []
    right_hand_features_all = []
    mo2cap2_body_features_all = []
    root_features_all = []

    left_hand_features_seq_all = []
    right_hand_features_seq_all = []
    mo2cap2_body_features_seq_all = []
    root_features_seq_all = []
    image_metas_seq_all = []


    for seq_id in tqdm(range(len(dataset))):
        data_i = dataset[seq_id]
        left_hand_features = data_i['left_hand_features']
        right_hand_features = data_i['right_hand_features']
        mo2cap2_body_features = data_i['mo2cap2_body_features']
        root_features = data_i['root_features']
        img_metas = data_i['img_metas'].data
        left_hand_features_all.extend(left_hand_features.numpy())
        right_hand_features_all.extend(right_hand_features.numpy())
        mo2cap2_body_features_all.extend(mo2cap2_body_features.numpy())
        root_features_all.extend(root_features.numpy())

        left_hand_features_seq_all.append(left_hand_features.numpy())
        right_hand_features_seq_all.append(right_hand_features.numpy())
        mo2cap2_body_features_seq_all.append(mo2cap2_body_features.numpy())
        root_features_seq_all.append(root_features.numpy())
        image_metas_seq_all.append(img_metas)


    left_hand_features_all = np.asarray(left_hand_features_all)
    right_hand_features_all = np.asarray(right_hand_features_all)
    mo2cap2_body_features_all = np.asarray(mo2cap2_body_features_all)
    root_features_all = np.asarray(root_features_all)

    left_hand_features_mean = np.mean(left_hand_features_all, axis=0)
    left_hand_features_std = np.std(left_hand_features_all, axis=0)
    right_hand_features_mean = np.mean(right_hand_features_all, axis=0)
    right_hand_features_std = np.std(right_hand_features_all, axis=0)

    mo2cap2_body_features_mean = np.mean(mo2cap2_body_features_all, axis=0)
    mo2cap2_body_features_std = np.std(mo2cap2_body_features_all, axis=0)

    root_features_mean = np.mean(root_features_all, axis=0)
    root_features_std = np.std(root_features_all, axis=0)

    print(f'left_hand_features_mean: {left_hand_features_mean}')
    print(f'left_hand_features_std: {left_hand_features_std}')
    print(f'right_hand_features_mean: {right_hand_features_mean}')
    print(f'right_hand_features_std: {right_hand_features_std}')
    print(f'mo2cap2_body_features_mean: {mo2cap2_body_features_mean}')
    print(f'mo2cap2_body_features_std: {mo2cap2_body_features_std}')
    print(f'root_features_mean: {root_features_mean}')
    print(f'root_features_std: {root_features_std}')

    mean_and_std = {
        'left_hand_features_mean': left_hand_features_mean,
        'left_hand_features_std': left_hand_features_std,
        'right_hand_features_mean': right_hand_features_mean,
        'right_hand_features_std': right_hand_features_std,
        'mo2cap2_body_features_mean': mo2cap2_body_features_mean,
        'mo2cap2_body_features_std': mo2cap2_body_features_std,
        'root_features_mean': root_features_mean,
        'root_features_std': root_features_std,
    }

    features_seqs = {
        'left_hand_features_seqs': left_hand_features_seq_all,
        'right_hand_features_seqs': right_hand_features_seq_all,
        'mo2cap2_body_features_seqs': mo2cap2_body_features_seq_all,
        'root_features_seqs': root_features_seq_all,
        'image_metas_seqs': image_metas_seq_all,
    }


    with open(save_mean_std_path, 'wb') as f:
        pickle.dump(mean_and_std, f)

    with open(save_data_path, 'wb') as f:
        pickle.dump(features_seqs, f)



if __name__ == '__main__':
    if os.name == 'nt':
        egobody_data_path = r'Z:\datasets04\static00\EgoBody'
        renderpeople_data_path = r'X:\ScanNet\work\synthetic_dataset_egofullbody\render_people_mixamo'
        smplx_model_dir = r'Z:\EgoMocap\work\EgocentricFullBody\human_models\smplx_new'
        save_dir = r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\diffusion_fullbody'
    else:
        egobody_data_path = '/CT/datasets04/static00/EgoBody'
        renderpeople_data_path = r'/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo'
        smplx_model_dir = '/CT/EgoMocap/work/EgocentricFullBody/human_models/smplx_new'
        save_dir = '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/diffusion_fullbody'

    calculate_egobody_mean_std()
