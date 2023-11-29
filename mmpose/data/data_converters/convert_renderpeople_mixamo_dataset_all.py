"""
this code is for converting the render people mixamo dataset sequences to one entire file.

This script is used for the complete sequence of mixamo dataset
"""
import os
import pickle
import sys

import cv2
import numpy as np
from natsort import natsorted

sys.path.append('../../..')

# from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_joint_names, mo2cap2_to_render_people_joints
from mmpose.data.keypoints_mapping.renderpeople import render_people_orginal_joint_names
from mmpose.utils.visualization.draw import draw_joints
from mmpose.utils.visualization.skeleton import Skeleton
from tqdm import tqdm


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
            r'Z:\EgoMocap\work\EgocentricFullBody\mmpose\utils\fisheye_camera\fisheye.calibration_01_12.json')
        # renderpeople to mo2cap2
        mesh = skeleton.joints_2_mesh(joint_3d_local)

        import open3d
        coor = open3d.geometry.TriangleMesh.create_coordinate_frame()
        open3d.visualization.draw_geometries([coor, mesh])

        # joint_2d = skeleton.camera.world2camera(joint_3d_local)
        # img = cv2.imread(image_path)
        # img = draw_joints(joint_2d, img)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

    return joint_3d_local


def get_renderpeople_joints(renderpeople_joints):
    joint_list = []
    for render_people_joint_name in render_people_orginal_joint_names:
        joint_loc = renderpeople_joints[render_people_joint_name]['location']
        joint_list.append(joint_loc)
    return np.asarray(joint_list)


def convert_renderpeople_mixamo_dataset(base_path, mode='train'):
    if mode == 'train':
        identity_name_list = [
                              'render_people_adanna',
                              ]
        # identity_name_list = ['render_people_claudia',
        #                       'render_people_eric',
        #                       'render_people_carla',
        #                       'render_people_adanna',
        #                       'render_people_amit',
        #                       'render_people_janna',
        #                       'render_people_joko',
        #                       'render_people_joyce',
        #                       'render_people_kyle',
        #                       'render_people_maya',
        #                       'render_people_rin',
        #                       'render_people_scott',
        #                       'render_people_serena',
        #                       'render_people_shawn',
        #                       ]
    else:
        identity_name_list = []
    seq_data = {
        'metainfo': {
            "name": "renderpeople_mixamo",
            'mode': mode,
            'identity_name_list': identity_name_list,
            "renderpeople_joint_name_list": render_people_orginal_joint_names,
            # "mo2cap2_joint_name_list": mo2cap2_joint_names,
        },
        'data_list': {

        }
    }
    for identity_name in identity_name_list:
        idendity_path = os.path.join(base_path, identity_name)
        out_identity_path = os.path.join(base_path, f'{identity_name}_joints_all.pkl')
        if os.path.exists(out_identity_path):
            print(f'{out_identity_path} exists!')
            with open(out_identity_path, 'rb') as f:
                data_identity = pickle.load(f)
            seq_data['data_list'][identity_name] = data_identity
            continue
        print(f'running: {idendity_path}')
        data_identity = {}
        for motion_name in tqdm(sorted(os.listdir(idendity_path))):
            motion_data = []
            motion_path = os.path.join(idendity_path, motion_name)
            joint_info_pkl = os.path.join(motion_path, 'joint_info_all.pkl')
            with open(joint_info_pkl, 'rb') as f:
                joint_info_data = pickle.load(f)
            camera_info_list = joint_info_data['camera_pose_list']
            body_pose_list = joint_info_data['body_pose_list']

            for i, (camera_info_i, body_pose_i) in enumerate(zip(camera_info_list, body_pose_list)):


                renderpeople_joints = get_renderpeople_joints(body_pose_i)
                renderpeople_local_joints = get_egocentric_pose(renderpeople_joints, camera_info_i, visualization=False)

                motion_data.append(
                    {
                        # 'joint_info': joint_info_i,
                        'camera_info': camera_info_i,
                        # 'mo2cap2_local_joints': mo2cap2_local_joints,
                        'renderpeople_local_joints': renderpeople_local_joints,
                        'renderpeople_joints': renderpeople_joints
                    }
                )
            data_identity[motion_name] = motion_data
        # save pkl for each identity

        with open(out_identity_path, 'wb') as f:
            pickle.dump(data_identity, f)
        seq_data['data_list'][identity_name] = data_identity
    # save pkl
    out_path = os.path.join(base_path, 'renderpeople_mixamo_labels_all.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(seq_data, f)


if __name__ == '__main__':
    base_path = r'/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo'
    # base_path = r'X:\ScanNet\work\synthetic_dataset_egofullbody\render_people_mixamo'
    convert_renderpeople_mixamo_dataset(base_path)
