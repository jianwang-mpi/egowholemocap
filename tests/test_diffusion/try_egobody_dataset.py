#  Copyright Jian Wang @ MPI-INF (c) 2023.
import os

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
                                       PreProcessHandMotion)
from mmpose.utils.visualization.draw import draw_keypoints_3d, draw_skeleton_with_chain
from mmpose.utils.visualization.skeleton import Skeleton

def try_egobody_dataset(seq_id):
    seq_len=196
    pipeline = [
        CalculateEgocentricCameraLocationFromSMPLX(random_camera_rotation=True, random_camera_translation=True),
        ConvertSMPLXOutputToEgocentricCameraLocation(),
        SplitEgoHandMotion(),
        PreProcessHandMotion(normalize=False),
        Collect(keys=['smplx_output', 'smplx_input',
                    'global_smplx_joints', 'global_smplx_vertices',
                      'ego_camera_rot', 'ego_camera_transl', 'ego_camera_transform',
                      'ego_smplx_joints', 'ego_smplx_vertices', 'processed_left_hand_keypoints_3d',
                      'processed_right_hand_keypoints_3d', 'json_path_calib'],
                meta_keys=[])
    ]


    dataset_cfg = dict(
        type='EgoBodyDataset',
        data_path=r'Z:\datasets04\static00\EgoBody',
        seq_len=seq_len,
        skip_frames=1,
        smplx_model_dir=r'Z:\EgoMocap\work\EgocentricFullBody\human_models\smplx_new',
        pipeline=pipeline,
        split_sequence=False,
        data_dirs=['smplx_camera_wearer_train'],
        test_mode=True)

    egobody_dataset = build_dataset(dataset_cfg)

    print(f'length of dataset is: {len(egobody_dataset)}')

    data_i = egobody_dataset[seq_id]

    smplx_output_i = data_i['smplx_output']
    smplx_input_i = data_i['smplx_input']
    ego_camera_rotation_i = data_i['ego_camera_rot']
    ego_camera_translation_i = data_i['ego_camera_transl']
    ego_camera_transform_i = data_i['ego_camera_transform']

    ego_smplx_joints = data_i['ego_smplx_joints']
    ego_smplx_vertices = data_i['ego_smplx_vertices']

    global_smplx_joints = data_i['global_smplx_joints']
    global_smplx_vertices = data_i['global_smplx_vertices']
    json_path_calib = data_i['json_path_calib']
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



    for frame_id in [0, 25, 50, 75, 100, 125, 150]:
        # visualize global smplx joints and vertices

        global_smplx_joints_mesh = draw_keypoints_3d(global_smplx_joints.numpy()[frame_id])
        world_coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

        global_smplx_mesh_i = open3d.geometry.TriangleMesh()
        global_smplx_mesh_i.vertices = open3d.utility.Vector3dVector(global_smplx_vertices[frame_id])
        global_smplx_mesh_i.triangles = open3d.utility.Vector3iVector(smplx_faces_i)
        global_smplx_mesh_i.compute_vertex_normals()
        open3d.visualization.draw_geometries([global_smplx_mesh_i, world_coord, scene_mesh])


        # visualize hands
        # left_hand_keypoints_3d = left_hand_diffusion_input[frame_id].numpy()
        # right_hand_keypoints_3d = right_hand_diffusion_input[frame_id].numpy()
        #
        # # left_hand_keypoints_3d[0] = np.array([0, 0, 0])
        # # right_hand_keypoints_3d[0] = np.array([0, 0, 0])
        #
        # left_hand_keypoints_3d[1:] += left_hand_keypoints_3d[0:1]
        # right_hand_keypoints_3d[1:] += right_hand_keypoints_3d[0:1]
        #
        # left_hand_mesh = draw_skeleton_with_chain(left_hand_keypoints_3d, mano_skeleton, keypoint_radius=0.01,
        #                                           line_radius=0.0025)
        # right_hand_mesh = draw_skeleton_with_chain(right_hand_keypoints_3d, mano_skeleton, keypoint_radius=0.01,
        #                                           line_radius=0.0025)
        #
        #
        #
        #
        # ego_smplx_joints_mesh = draw_keypoints_3d(ego_smplx_joints.numpy()[frame_id])
        # world_coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
        #
        # ego_smplx_mesh_i = open3d.geometry.TriangleMesh()
        # ego_smplx_mesh_i.vertices = open3d.utility.Vector3dVector(ego_smplx_vertices[frame_id])
        # ego_smplx_mesh_i.triangles = open3d.utility.Vector3iVector(smplx_faces_i)
        # ego_smplx_mesh_i.compute_vertex_normals()
        #
        # open3d.visualization.draw_geometries([left_hand_mesh, right_hand_mesh, ego_smplx_mesh_i])
        #
        # open3d.visualization.draw_geometries([world_coord, ego_smplx_joints_mesh, ego_smplx_mesh_i ])

        # smplx_mesh_i = open3d.geometry.TriangleMesh()
        # smplx_mesh_i.vertices = open3d.utility.Vector3dVector(smplx_vertices_i[frame_id])
        # smplx_mesh_i.triangles = open3d.utility.Vector3iVector(smplx_faces_i)
        # smplx_mesh_i.compute_vertex_normals()
        #
        #
        # # visualize the egocentric camera
        # #
        # ego_camera_coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # ego_camera_coord = ego_camera_coord.transform(ego_camera_transform_i[frame_id])
        # open3d.visualization.draw_geometries([smplx_mesh_i, ego_camera_coord])
        #
        # # convert smplx joints to camera location
        # smplx_joints_i = smplx_output_i.joints.cpu().numpy()

        # # smplx_joints_i = smplx_joints_i[frame_id]
        # smplx_joints_i = np.concatenate([smplx_joints_i, np.ones((smplx_joints_i.shape[0],
        #                                                           smplx_joints_i.shape[1], 1))], axis=2)
        # matrix = np.linalg.inv(ego_camera_transform_i)
        # smplx_joints_i = smplx_joints_i.reshape(-1, 4)
        # smplx_joints_i = np.matmul(matrix, smplx_joints_i.T).T
        # smplx_joints_i = smplx_joints_i[:, :3]
        # smplx_joints_i = smplx_joints_i.reshape(seq_len, -1, 3)




if __name__ == '__main__':
    try_egobody_dataset(0)

