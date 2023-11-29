#  Copyright Jian Wang @ MPI-INF (c) 2023.
#  Copyright Jian Wang @ MPI-INF (c) 2023.
import os
import pickle

import numpy as np
import open3d
import smplx

from mmpose.data.keypoints_mapping.mano import mano_skeleton
from mmpose.datasets.builder import build_dataset
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
from mmpose.datasets.pipelines import (Collect,
                                       PreProcessHandMotion, AlignGlobalSMPLXJoints, SplitGlobalSMPLXJoints,
                                       PreProcessMo2Cap2BodyMotion)
from mmpose.utils.visualization.draw import draw_keypoints_3d, draw_skeleton_with_chain


def try_interhand_dataset(seq_id):
    seq_len = 196
    pipeline = [
        Collect(keys=['global_hand_joints'],
                meta_keys=[])
    ]

    dataset_cfg = dict(
        type='EgoBodyDataset',
        data_path=r'Z:\datasets04\static00\EgoBody',
        seq_len=seq_len,
        skip_frames=seq_len,
        smplx_model_dir=r'Z:\EgoMocap\work\EgocentricFullBody\human_models\smplx_new',
        pipeline=pipeline,
        split_sequence=False,
        data_dirs=['smplx_camera_wearer_val'],
        test_mode=True)

    egobody_dataset = build_dataset(dataset_cfg)

    print(f'length of dataset is: {len(egobody_dataset)}')

    data_i = egobody_dataset[seq_id]

    smplx_output_i = data_i['smplx_output']
    smplx_input_i = data_i['smplx_input']

    global_smplx_joints = data_i['global_smplx_joints']
    global_smplx_vertices = data_i['global_smplx_vertices']
    json_path_calib = data_i['json_path_calib']

    aligned_smplx_joints = data_i['aligned_smplx_joints']

    mo2cap2_features = data_i['mo2cap2_body_features']
    left_hand_features = data_i['left_hand_features']
    right_hand_features = data_i['right_hand_features']

    # recover from mean and std
    with open(r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\egobody\global_aligned_mean_std.pkl', 'rb') as f:
        global_aligned_mean_std = pickle.load(f)
    mo2cap2_body_mean = global_aligned_mean_std['mo2cap2_body_mean']
    mo2cap2_body_std = global_aligned_mean_std['mo2cap2_body_std']
    mo2cap2_features = mo2cap2_features * mo2cap2_body_std + mo2cap2_body_mean
    left_hand_mean = global_aligned_mean_std['left_hand_mean']
    left_hand_std = global_aligned_mean_std['left_hand_std']
    left_hand_features = left_hand_features * left_hand_std + left_hand_mean
    right_hand_mean = global_aligned_mean_std['right_hand_mean']
    right_hand_std = global_aligned_mean_std['right_hand_std']
    right_hand_features = right_hand_features * right_hand_std + right_hand_mean

    scene_mesh_dir = r'\\winfs-inf\CT\datasets04\static00\EgoBody\scene_mesh'
    scene_mesh_name = os.path.split(json_path_calib)[1].split('.')[0]
    scene_mesh_path = os.path.join(scene_mesh_dir, scene_mesh_name, scene_mesh_name + '.obj')
    scene_mesh = open3d.io.read_triangle_mesh(scene_mesh_path)

    left_hand_diffusion_input = data_i['processed_left_hand_keypoints_3d']
    right_hand_diffusion_input = data_i['processed_right_hand_keypoints_3d']

    print(smplx_input_i.keys())

    # visualize the smplx mesh and egocentric camera
    smplx_model = smplx.create(r'Z:\EgoMocap\work\EgocentricFullBody\human_models\smplx_new\SMPLX_NEUTRAL.npz',
                               model_type='smplx', num_betas=10)

    smplx_vertices_i = smplx_output_i.vertices.cpu().numpy()
    smplx_faces_i = smplx_model.faces

    mo2cap2_idx, smplx_idx = dset_to_body_model(model_type='smplx', dset='mo2cap2')

    world_coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    for frame_id in range(0, 100, 10):
        mo2cap2_feature = mo2cap2_features[frame_id].numpy()
        left_hand_feature = left_hand_features[frame_id].numpy()
        right_hand_feature = right_hand_features[frame_id].numpy()

        mo2cap2_feature = mo2cap2_feature.reshape(15, 3)
        left_hand_feature = left_hand_feature.reshape(21, 3)
        left_hand_feature[0] *= 0
        right_hand_feature = right_hand_feature.reshape(21, 3)
        right_hand_feature[0] *= 0

        from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
        aligned_mo2cap2_joints_mesh = draw_skeleton_with_chain(mo2cap2_feature, mo2cap2_chain)
        coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        open3d.visualization.draw_geometries([aligned_mo2cap2_joints_mesh, coord])

        aligned_left_hand_mesh = draw_skeleton_with_chain(left_hand_feature, mano_skeleton, keypoint_radius=0.01,
                                                          line_radius=0.0025)
        aligned_right_hand_mesh = draw_skeleton_with_chain(right_hand_feature, mano_skeleton, keypoint_radius=0.01,
                                                          line_radius=0.0025)
        open3d.visualization.draw_geometries([aligned_left_hand_mesh, coord])
        open3d.visualization.draw_geometries([aligned_right_hand_mesh, coord])

        smplx_joint = aligned_smplx_joints.numpy()[frame_id]
        mo2cap2_joint = np.zeros((15, 3))
        mo2cap2_joint[mo2cap2_idx] = smplx_joint[smplx_idx]

        from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
        smplx_keypoints_mesh = draw_keypoints_3d(smplx_joint)
        aligned_mo2cap2_joints_mesh = draw_skeleton_with_chain(mo2cap2_joint, mo2cap2_chain)
        coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        open3d.visualization.draw_geometries([aligned_mo2cap2_joints_mesh, coord, smplx_keypoints_mesh])
        world_coord += aligned_mo2cap2_joints_mesh
    open3d.visualization.draw_geometries([world_coord])


if __name__ == '__main__':
    try_interhand_dataset(5)
