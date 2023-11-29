"""
this code is for converting the render people mixamo dataset and change it to the
"""
import os
import pickle
import sys
import smplx
import cv2
import numpy as np
from natsort import natsorted

sys.path.append('../../..')

from mmpose.data.keypoints_mapping.smplh import smplh_joint_names
from mmpose.data.keypoints_mapping.smpl import smpl_joint_names
from mmpose.data.keypoints_mapping.smplx import smplx_joint_names
from mmpose.utils.visualization.draw import draw_joints
from mmpose.utils.visualization.skeleton import Skeleton
from tqdm import tqdm
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model


def get_egocentric_pose(joint_3d, camera_matrix, visualization=False, image_path=None):
    blender_to_cv = np.array([[1, 0, 0],
                              [0, -1, 0],
                              [0, 0, -1]])
    camera_t = camera_matrix[:3, 3]
    camera_R = camera_matrix[:3, :3]
    camera_R = camera_R.dot(blender_to_cv)
    joint_3d = joint_3d - camera_t
    joint_3d_local = camera_R.T.dot(joint_3d.T).T

    if visualization:
        skeleton = Skeleton(
            r'Z:\EgoMocap\work\ViTEgocentricFullBody\egofullbody\utils\cameras\fisheye.calibration_01_12.json')
        # mesh = skeleton.joints_2_mesh(joint_3d_local)
        #
        # coor = open3d.geometry.TriangleMesh.create_coordinate_frame()
        # open3d.visualization.draw_geometries([coor, mesh])

        joint_2d = skeleton.camera.world2camera(joint_3d_local)
        img = cv2.imread(image_path)
        img = draw_joints(joint_2d, img)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    return joint_3d_local


def convert_renderpeople_mixamo_dataset(base_path, body_model_type='smplh', mode='train'):
    if body_model_type == 'smplh':
        joint_name_list = smplh_joint_names
    elif body_model_type == 'smpl':
        joint_name_list = smpl_joint_names
    else:
        joint_name_list = smplx_joint_names
    if mode == 'train':
        data_name_list = ['GRAB/s1',
                              'TCD_handMocap/ExperimentDatabase',
                              ]
    else:
        data_name_list = []
    seq_data = {
        'metainfo': {
            "name": "renderpeople_mixamo",
            'mode': mode,
            'data_name_list': data_name_list,
            "joint_name_list": joint_name_list,
        },
        'data_list': [

        ]
    }
    for data_name in data_name_list:
        dataset_name, dataset_seq_name = os.path.split(data_name)
        data_path = os.path.join(base_path, data_name)
        out_seq_path = os.path.join(base_path, f'{dataset_name}_{dataset_seq_name}.pkl')
        if os.path.exists(out_seq_path):
            print(f'{out_seq_path} exists!')
            with open(out_seq_path, 'rb') as f:
                data_identity = pickle.load(f)
            seq_data['data_list'].extend(data_identity)
            continue
        print(f'running: {data_path}')
        data_identity = []
        for motion_name in tqdm(sorted(os.listdir(data_path))):
            motion_path = os.path.join(data_path, motion_name)
            motion_relative_path = os.path.join(data_name, motion_name)
            meta_data_pkl = os.path.join(motion_path, 'metadata.pkl')
            if os.path.exists(meta_data_pkl):
                with open(meta_data_pkl, 'rb') as f:
                    meta_data = pickle.load(f)
            else:
                continue

            smplh_params = meta_data['smplh_params']
            joints = meta_data['joints']

            image_dir = os.path.join(motion_path, 'img')
            imagedir_relative_dir = os.path.join(motion_relative_path, 'img')
            depthsegdir_relative_dir = os.path.join(motion_relative_path, 'depth_seg')

            image_name_list = natsorted(os.listdir(image_dir))
            assert len(image_name_list) == len(joints)

            for i, image_name in enumerate(image_name_list):
                image_id = int(os.path.splitext(image_name)[0])
                assert i == (image_id - 1) / 5
                depth_seg_dir = '%03d' % image_id

                image_relative_path = os.path.join(imagedir_relative_dir, image_name)
                seg_relative_path = os.path.join(depthsegdir_relative_dir, depth_seg_dir, 'Seg0001.jpg')
                depth_relative_path = os.path.join(depthsegdir_relative_dir, depth_seg_dir, 'Depth0001.exr')
                joint_i = joints[i]
                camera_info_i = camera_info[i]

                mo2cap2_joints = get_mo2cap2_joints_from_renderpeople_joints(joint_info_i)
                mo2cap2_local_joints = get_egocentric_pose(mo2cap2_joints, camera_info_i, visualization=False,
                                                           image_path=os.path.join(base_path, image_relative_path))

                renderpeople_joints = get_renderpeople_joints(joint_info_i)
                renderpeople_local_joints = get_egocentric_pose(renderpeople_joints, camera_info_i, visualization=False)

                data_identity.append(
                    {
                        'img_path': image_relative_path,
                        'seg_path': seg_relative_path,
                        'depth_path': depth_relative_path,
                        # 'joint_info': joint_info_i,
                        'camera_info': camera_info_i,
                        'mo2cap2_local_joints': mo2cap2_local_joints,
                        'renderpeople_local_joints': renderpeople_local_joints
                    }
                )
        # save pkl for each identity

        with open(out_identity_path, 'wb') as f:
            pickle.dump(data_identity, f)
        seq_data['data_list'].extend(data_identity)
    # save pkl
    out_path = os.path.join(base_path, 'smplh_amass_labels.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(seq_data, f)


if __name__ == '__main__':
    # base_path = '/HPS/ScanNet/work/synthetic_dataset_egofullbody/smplh_amass'
    base_path = 'X:/ScanNet/work/synthetic_dataset_egofullbody/smplh_amass'
    convert_renderpeople_mixamo_dataset(base_path)
