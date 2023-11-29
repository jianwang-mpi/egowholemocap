"""
this code is for converting the render people mixamo dataset and change it to the
"""
import os
import pickle
import sys

import open3d
import smplx
import cv2
import numpy as np
import torch
from natsort import natsorted
from scipy.spatial.transform import Rotation

sys.path.append('../../..')

from mmpose.data.keypoints_mapping.smplh import smplh_joint_names
from mmpose.data.keypoints_mapping.smpl import smpl_joint_names
from mmpose.data.keypoints_mapping.smplx import smplx_joint_names
from mmpose.utils.visualization.draw import draw_joints
from mmpose.utils.visualization.skeleton import Skeleton
from tqdm import tqdm
import os.path as osp
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
from mmpose.utils.blender.blender_camera import get_cv_rt_from_blender

def get_camera_matrix(camera_info):
    camera_blender_loc = camera_info['loc']
    camera_blender_rot = camera_info['rot']

    T_world2cv, R_world2cv, mat = get_cv_rt_from_blender(camera_blender_loc, camera_blender_rot)
    return mat

def get_egocentric_pose(joint_3d, camera_info, visualization=False, image_path=None):
    camera_t = camera_info['loc']
    camera_blender_rot = camera_info['rot']
    camera_R = Rotation.from_euler('xyz', camera_blender_rot, degrees=False).as_matrix()

    blender_to_cv = np.array([[1, 0, 0],
                              [0, -1, 0],
                              [0, 0, -1]])
    camera_R = camera_R.dot(blender_to_cv)
    joint_3d = joint_3d - camera_t
    joint_3d_local = camera_R.T.dot(joint_3d.T).T

    if visualization:
        skeleton = Skeleton(
            r'Z:\EgoMocap\work\ViTEgocentricFullBody\egofullbody\utils\cameras\fisheye.calibration_01_12.json')
        skeleton = Skeleton(None)
        mesh = skeleton.joints_2_mesh(joint_3d_local)

        coor = open3d.geometry.TriangleMesh.create_coordinate_frame()
        open3d.visualization.draw_geometries([coor, mesh])

        joint_2d = skeleton.camera.world2camera(joint_3d_local)[0] / 2
        img = cv2.imread(image_path)
        img = draw_joints(joint_2d, img)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    return joint_3d_local

def get_smpl_joints(smpl_model, pose_list, beta_list, trans_list):
    poses = torch.asarray(np.asarray(pose_list)).float()
    global_orient = poses[:, :3]
    body_pose = poses[:, 3:]
    betas = torch.asarray(np.asarray(beta_list)).float()
    transl = torch.asarray(np.asarray(trans_list)).float()

    smpl_body = smpl_model(**{
        'global_orient': global_orient,
        'body_pose': body_pose,
        'betas': betas,
        'transl': transl
    })
    smpl_body_joints = smpl_body.joints
    return smpl_body_joints.detach().cpu().numpy()

def convert_smpl_dataset(base_path, mode='train'):
    if mode == 'train':
        # get data name list from base path
        scene_name_list = os.listdir(base_path)
        scene_name_list = natsorted([d for d in scene_name_list if osp.isdir(osp.join(base_path, d))])
    else:
        scene_name_list = []
    seq_data = {
        'metainfo': {
            "name": "smpl_matterport",
            'mode': mode,
            "joint_name_list": smpl_joint_names,
        },
        'data_list': [

        ]
    }
    for scene_name in scene_name_list:
        scene_path = os.path.join(base_path, scene_name)
        motion_name_list = os.listdir(scene_path)
        motion_name_list = natsorted([d for d in motion_name_list if osp.isdir(osp.join(scene_path, d))])
        out_scene_path = os.path.join(base_path, f'{scene_name}.pkl')
        if os.path.exists(out_scene_path):
            print(f'{out_scene_path} exists!')
            with open(out_scene_path, 'rb') as f:
                data_scene = pickle.load(f)
            seq_data['data_list'].extend(data_scene)
            continue
        print(f'running: {scene_path}')
        data_scene = []
        for motion_name in tqdm(motion_name_list):
            motion_path = os.path.join(scene_path, motion_name)
            motion_relative_path = os.path.join(scene_name, motion_name)
            meta_data_npy = os.path.join(motion_path, 'metadata.npy')
            if os.path.exists(meta_data_npy):
                meta_data = np.load(meta_data_npy, allow_pickle=True).item()
            else:
                continue

            # print(type(meta_data))
            body_data = meta_data['body_data']
            gender = meta_data['gender']
            camera_data = meta_data['camera_data']

            if gender == 'Female':
                smpl_model = smplx.create('/CT/EgoMocap/work/EgocentricFullBody/models/smpl_numpy/smpl_numpy_f.pkl', model_type='smpl', gender='female')
            else:
                smpl_model = smplx.create('/CT/EgoMocap/work/EgocentricFullBody/models/smpl_numpy/smpl_numpy_m.pkl', model_type='smpl', gender='male')

            image_dir = os.path.join(motion_path, 'img')
            imagedir_relative_dir = os.path.join(motion_relative_path, 'img')
            depthdir_relative_dir = os.path.join(motion_relative_path, 'depth')

            image_name_list = natsorted(os.listdir(image_dir))
            assert len(image_name_list) == len(body_data)

            pose_list = [body_data_i['pose'] for body_data_i in body_data]
            shape_list = [body_data_i['shape'] for body_data_i in body_data]
            trans_list = [body_data_i['trans'] for body_data_i in body_data]

            smpl_joint_list = get_smpl_joints(smpl_model, pose_list, shape_list, trans_list)

            for i, image_name in enumerate(image_name_list):
                image_id = int(os.path.splitext(image_name)[0])
                assert i == image_id / 4

                camera_info_i = camera_data[i]
                smpl_joints_i = smpl_joint_list[i]

                depth_dir = '%03d' % image_id

                image_relative_path = os.path.join(imagedir_relative_dir, image_name)
                depth_relative_path = os.path.join(depthdir_relative_dir, depth_dir, 'Image0001.exr')

                # camera_matrix_i = get_camera_matrix(camera_info_i)

                smpl_local_joints = get_egocentric_pose(smpl_joints_i, camera_info_i, visualization=False,
                                                           image_path=os.path.join(base_path, image_relative_path))

                data_scene.append(
                    {
                        'img_path': image_relative_path,
                        'depth_path': depth_relative_path,
                        # 'joint_info': joint_info_i,
                        'camera_info': camera_info_i,
                        'smpl_local_joints': smpl_local_joints,
                        'smpl_params': body_data[i]
                    }
                )
        # save pkl for each identity

        with open(out_scene_path, 'wb') as f:
            pickle.dump(data_scene, f)
        seq_data['data_list'].extend(data_scene)
    # save pkl
    out_path = os.path.join(base_path, 'smplh_amass_labels.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(seq_data, f)


if __name__ == '__main__':
    # base_path = '/HPS/ScanNet/work/synthetic_dataset_egofullbody/smplh_amass'
    base_path = '/HPS/EgoSyn/work/synthetic/depth_matterport_seg_seq'
    convert_smpl_dataset(base_path)
