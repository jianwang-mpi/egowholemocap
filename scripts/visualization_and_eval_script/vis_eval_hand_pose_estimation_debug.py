#  Copyright Jian Wang @ MPI-INF (c) 2023.

import os
import pickle
from copy import deepcopy

import numpy as np
import open3d
import torch
from tqdm import tqdm

from mmpose.datasets.datasets.egocentric.hand_joint_eval import hand_keypoints_3d_to_fisheye_camera_space
from mmpose.models.ego_hand_pose_estimation.utils.human_models import mano
from mmpose.utils.geometry_utils.geometry_utils_torch import convert_points_to_homogeneous
from mmpose.utils.visualization.hand_skeleton import HandSkeleton


def get_bbox_center(bbox):
    # bbox shape: (n, 4)
    bbox_center = np.empty((bbox.shape[0], 2))
    bbox_center[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2
    bbox_center[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2
    return bbox_center


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0] + 1) + '/' + str(f[i][0] + 1) + ' ' + str(f[i][1] + 1) + '/' + str(
            f[i][1] + 1) + ' ' + str(f[i][2] + 1) + '/' + str(f[i][2] + 1) + '\n')
    obj_file.close()


def eval_hand_detection(drawbbox=False):
    hand_skeleton = HandSkeleton('mano')
    fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
    result_path = r'/CT/EgoMocap/work/EgocentricFullBody/work_dirs/hands4whole_test/results.pkl'

    out_dir = '/CT/EgoMocap/work/EgocentricFullBody/vis_results/hand_pose_estimation'
    os.makedirs(out_dir, exist_ok=True)

    with open(result_path, 'rb') as f:
        result_data = pickle.load(f)

    mpjpe_list = []
    for i in tqdm(range(len(result_data))):
        result_data_i = result_data[i]
        image_file_list = []
        image_name_list = []
        image_id_list = []
        keypoints_gt_list = []
        left_hands_mano_pose_list = result_data_i['left_hands_preds']['mano_pose']
        left_hands_mano_shape_list = result_data_i['left_hands_preds']['mano_shape']
        right_hands_mano_pose_list = result_data_i['right_hands_preds']['mano_pose']
        right_hands_mano_shape_list = result_data_i['right_hands_preds']['mano_shape']

        # get predicted 3d pose

        right_hands_mano = mano.layer['right'](global_orient=right_hands_mano_pose_list[:, :3],
                                         hand_pose=right_hands_mano_pose_list[:, 3:],
                                         betas=right_hands_mano_shape_list)

        left_hands_mano = mano.layer['right'](global_orient=left_hands_mano_pose_list[:, :3],
                                               hand_pose=left_hands_mano_pose_list[:, 3:],
                                               betas=left_hands_mano_shape_list)


        right_vertices_mano = right_hands_mano.vertices
        left_vertices_mano = left_hands_mano.vertices
        left_vertices_mano[:, :, 0] *= -1

        right_joint_mano = torch.bmm(torch.from_numpy(mano.joint_regressor)[None, :, :].repeat(right_vertices_mano.shape[0], 1, 1),
                              right_vertices_mano)

        left_joint_mano = torch.bmm(torch.from_numpy(mano.joint_regressor)[None, :, :].repeat(left_vertices_mano.shape[0], 1, 1),
                              left_vertices_mano)

        left_joint_cam = result_data_i['left_hands_preds']['joint_cam']
        right_joint_cam = result_data_i['right_hands_preds']['joint_cam']

        left_hand_keypoints_3d_gt = []
        right_hand_keypoints_3d_gt = []

        left_hand_transform_list = []
        right_hand_transform_list = []

        img_meta_list = result_data_i['img_metas']
        for img_meta_item in img_meta_list:
            image_file = img_meta_item['image_file']
            image_file_list.append(image_file)
            image_name_list.append(os.path.basename(image_file))
            image_id_list.append(image_file.split('/')[-3])

            left_hand_keypoints_3d = img_meta_item['left_hand_keypoints_3d']
            left_hand_keypoints_3d_gt.append(left_hand_keypoints_3d)
            right_hand_keypoints_3d = img_meta_item['right_hand_keypoints_3d']
            right_hand_keypoints_3d_gt.append(right_hand_keypoints_3d)

            left_transform_matrix = img_meta_item['left_hand_transform']
            right_transform_matrix = img_meta_item['right_hand_transform']

            left_hand_transform_list.append(left_transform_matrix)
            right_hand_transform_list.append(right_transform_matrix)
        left_hand_keypoints_3d_gt = np.asarray(left_hand_keypoints_3d_gt)
        right_hand_keypoints_3d_gt = np.asarray(right_hand_keypoints_3d_gt)
        left_hand_transform_list = np.asarray(left_hand_transform_list)
        right_hand_transform_list = np.asarray(right_hand_transform_list)


        # select one of the hand
        hand_index = 0

        right_hand_transform_i = right_hand_transform_list[hand_index]
        right_joint_mano_i = right_joint_mano[hand_index].numpy()
        right_joint_cam_i = right_joint_cam[hand_index]
        right_hand_keypoints_3d_gt_i = right_hand_keypoints_3d_gt[hand_index]
        right_joint_mano_fisheye_coord = (right_hand_transform_i[:3, :3] @ right_joint_mano_i.T).T


        # right_hand_keypoints_3d_gt_i = np.concatenate([right_hand_keypoints_3d_gt_i[0:1],
        #                                                right_hand_keypoints_3d_gt_i[17:21],
        #                                                right_hand_keypoints_3d_gt_i[1:5],
        #                                                right_hand_keypoints_3d_gt_i[5:9],
        #                                                right_hand_keypoints_3d_gt_i[13:17],
        #                                                right_hand_keypoints_3d_gt_i[9:13]
        #                                                ])
        # put the pose to the center of space
        right_hand_keypoints_3d_gt_i -= np.mean(right_hand_keypoints_3d_gt_i, axis=0)
        right_joint_mano_fisheye_coord -= np.mean(right_joint_mano_fisheye_coord, axis=0)

        pred_keypoints_mesh = hand_skeleton.joints_2_mesh(right_joint_mano_fisheye_coord)
        gt_keypoints_mesh = hand_skeleton.joints_2_mesh(right_hand_keypoints_3d_gt_i,
                                                        joint_color=(1, 0, 0),
                                                        bone_color=(1, 0, 0))
        mesh_out_path = os.path.join(out_dir, f'rhand_transformed_{image_id_list[hand_index]}_{image_name_list[hand_index]}.ply')
        open3d.io.write_triangle_mesh(mesh_out_path, pred_keypoints_mesh + gt_keypoints_mesh)

        # deal with left hand
        left_hand_transform_i = left_hand_transform_list[hand_index]
        left_joint_mano_i = left_joint_mano[hand_index].numpy()
        left_hand_keypoints_3d_gt_i = left_hand_keypoints_3d_gt[hand_index]
        left_joint_mano_fisheye_coord = (left_hand_transform_i[:3, :3] @ left_joint_mano_i.T).T


        # put the pose to the center of space
        left_hand_keypoints_3d_gt_i -= np.mean(left_hand_keypoints_3d_gt_i, axis=0)
        left_joint_mano_fisheye_coord -= np.mean(left_joint_mano_fisheye_coord, axis=0)

        pred_keypoints_mesh = hand_skeleton.joints_2_mesh(left_joint_mano_fisheye_coord)
        gt_keypoints_mesh = hand_skeleton.joints_2_mesh(left_hand_keypoints_3d_gt_i,
                                                        joint_color=(1, 0, 0),
                                                        bone_color=(1, 0, 0))
        mesh_out_path = os.path.join(out_dir,
                                     f'lhand_transformed_{image_id_list[hand_index]}_{image_name_list[hand_index]}.ply')
        open3d.io.write_triangle_mesh(mesh_out_path, pred_keypoints_mesh + gt_keypoints_mesh)


if __name__ == '__main__':
    eval_hand_detection(drawbbox=True)
