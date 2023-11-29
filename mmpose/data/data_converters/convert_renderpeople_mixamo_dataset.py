"""
this code is for converting the render people mixamo dataset and change it to the
"""
import os
import pickle
import sys

import cv2
import numpy as np
from natsort import natsorted

sys.path.append('../../..')

from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_joint_names, mo2cap2_to_render_people_joints
from mmpose.data.keypoints_mapping.renderpeople import render_people_joint_names
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


def get_mo2cap2_joints_from_renderpeople_joints(renderpeople_joints):
    joint_list = []
    for mo2cap2_joint in mo2cap2_joint_names:
        render_people_joint = mo2cap2_to_render_people_joints[mo2cap2_joint]
        joint_loc = renderpeople_joints[render_people_joint]['location']
        joint_list.append(joint_loc)
    return np.asarray(joint_list)


def get_renderpeople_joints(renderpeople_joints):
    joint_list = []
    for render_people_joint_name in render_people_joint_names:
        joint_loc = renderpeople_joints[render_people_joint_name]['location']
        joint_list.append(joint_loc)
    return np.asarray(joint_list)


def convert_renderpeople_mixamo_dataset(base_path, mode='train'):
    if mode == 'train':
        identity_name_list = ['render_people_claudia',
                              'render_people_eric',
                              'render_people_carla',
                              'render_people_adanna',
                              'render_people_amit',
                              'render_people_janna',
                              'render_people_joko',
                              'render_people_joyce',
                              'render_people_kyle',
                              'render_people_maya',
                              'render_people_rin',
                              'render_people_scott',
                              'render_people_serena',
                              'render_people_shawn',
                              ]
    else:
        identity_name_list = []
    seq_data = {
        'metainfo': {
            "name": "renderpeople_mixamo",
            'mode': mode,
            'identity_name_list': identity_name_list,
            "renderpeople_joint_name_list": render_people_joint_names,
            "mo2cap2_joint_name_list": mo2cap2_joint_names,
        },
        'data_list': [

        ]
    }
    for identity_name in identity_name_list:
        idendity_path = os.path.join(base_path, identity_name)
        out_identity_path = os.path.join(base_path, f'{identity_name}.pkl')
        if os.path.exists(out_identity_path):
            print(f'{out_identity_path} exists!')
            with open(out_identity_path, 'rb') as f:
                data_identity = pickle.load(f)
            seq_data['data_list'].extend(data_identity)
            continue
        print(f'running: {idendity_path}')
        data_identity = []
        for motion_name in tqdm(sorted(os.listdir(idendity_path))):
            motion_path = os.path.join(idendity_path, motion_name)
            motion_relative_path = os.path.join(identity_name, motion_name)
            joint_info_pkl = os.path.join(motion_path, 'joint_info.pkl')
            camera_info_pkl = os.path.join(motion_path, 'ego_camera_info.pkl')
            if os.path.exists(joint_info_pkl) and os.path.exists(camera_info_pkl):
                with open(joint_info_pkl, 'rb') as f:
                    joint_info = pickle.load(f)
                with open(camera_info_pkl, 'rb') as f:
                    camera_info = pickle.load(f)
            else:
                body_pose_info_pkl = os.path.join(motion_path, 'body_pose_info.pkl')
                with open(body_pose_info_pkl, 'rb') as f:
                    body_pose_info_pkl = pickle.load(f)
                joint_info = [bp_info['joint_info'] for bp_info in body_pose_info_pkl]
                camera_info = [bp_info['camera_info'] for bp_info in body_pose_info_pkl]

            image_dir = os.path.join(motion_path, 'img')
            imagedir_relative_dir = os.path.join(motion_relative_path, 'img')
            segdir_relative_dir = os.path.join(motion_relative_path, 'seg')
            depthdir_relative_dir = os.path.join(motion_relative_path, 'depth')

            image_name_list = natsorted(os.listdir(image_dir))
            assert len(image_name_list) == len(joint_info) and len(image_name_list) == len(camera_info)

            for i, image_name in enumerate(image_name_list):
                image_id = int(os.path.splitext(image_name)[0])
                assert i == (image_id - 1) / 5
                seg_name = '%04d.jpg' % image_id
                depth_name = '%04d.exr' % image_id

                image_relative_path = os.path.join(imagedir_relative_dir, image_name)
                seg_relative_path = os.path.join(segdir_relative_dir, seg_name)
                depth_relative_path = os.path.join(depthdir_relative_dir, depth_name)
                joint_info_i = joint_info[i]
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
    out_path = os.path.join(base_path, 'renderpeople_mixamo_labels.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(seq_data, f)


if __name__ == '__main__':
    base_path = r'/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo'
    # base_path = r'X:\ScanNet\work\synthetic_dataset_egofullbody\render_people_mixamo'
    convert_renderpeople_mixamo_dataset(base_path)
