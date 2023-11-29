#  Copyright Jian Wang @ MPI-INF (c) 2023.
import pickle
from copy import deepcopy

import numpy as np
import open3d
import torch
import os

from mmpose.data.keypoints_mapping.mano import mano_skeleton
from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
from mmpose.utils.visualization.draw import draw_skeleton_with_chain


def visualize_joints(mo2cap2_joints, left_hand_joints, right_hand_joints, image_file_path_list):
    assert len(mo2cap2_joints) == len(left_hand_joints) == len(right_hand_joints) == len(image_file_path_list)

    for i in range(19300, len(mo2cap2_joints), 10):
        print(image_file_path_list[i])
        pred_left_hand = left_hand_joints[i]
        pred_right_hand = right_hand_joints[i]
        pred_body_pose = mo2cap2_joints[i]
        pred_right_hand += pred_body_pose[3] - pred_right_hand[0]
        pred_left_hand += pred_body_pose[6] - pred_left_hand[0]

        left_hand_mesh = draw_skeleton_with_chain(pred_left_hand, mano_skeleton, keypoint_radius=0.01,
                                                  line_radius=0.0025)
        right_hand_mesh = draw_skeleton_with_chain(pred_right_hand, mano_skeleton, keypoint_radius=0.01,
                                                   line_radius=0.0025)

        mo2cap2_mesh = draw_skeleton_with_chain(pred_body_pose, mo2cap2_chain)
        open3d.visualization.draw_geometries([mo2cap2_mesh, left_hand_mesh, right_hand_mesh])

def visualize_result(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    pred_joints_3d_list = []
    pred_left_hand_joint_3d_list = []
    pred_right_hand_joint_3d_list = []
    gt_joints_3d_list = []
    image_file_path_list = []
    for pred in data:
        pred_joints_3d_item = pred['body_pose_results']['keypoints_pred']

        pred_left_hand_joint_item = pred['left_hands_preds']['joint_cam_transformed']
        pred_right_hand_joint_item = pred['right_hands_preds']['joint_cam_transformed']

        if torch.is_tensor(pred_joints_3d_item):
            pred_joints_3d_item = pred_joints_3d_item.cpu().numpy()
        if torch.is_tensor(pred_left_hand_joint_item):
            pred_left_hand_joint_item = pred_left_hand_joint_item.cpu().numpy()
            pred_right_hand_joint_item = pred_right_hand_joint_item.cpu().numpy()

        pred_joints_3d_list.extend(pred_joints_3d_item)
        pred_left_hand_joint_3d_list.extend(pred_left_hand_joint_item)
        pred_right_hand_joint_3d_list.extend(pred_right_hand_joint_item)
        img_meta_list = pred['img_metas']
        for img_meta_item in img_meta_list:
            gt_joints_3d_item = img_meta_item['keypoints_3d']
            image_file_path = img_meta_item['image_file']
            gt_joints_3d_list.append(gt_joints_3d_item)
            image_file_path_list.append(image_file_path)

    gt_joints_3d_list = np.array(gt_joints_3d_list)
    pred_joints_3d_list = np.array(pred_joints_3d_list)
    pred_left_hand_joint_3d_list = np.array(pred_left_hand_joint_3d_list)
    pred_right_hand_joint_3d_list = np.array(pred_right_hand_joint_3d_list)
    # convert from model format to mo2cap2 format
    original_gt_joint_3d_list = deepcopy(gt_joints_3d_list)
    print(pred_joints_3d_list.shape)

    visualize_joints(pred_joints_3d_list, pred_left_hand_joint_3d_list, pred_right_hand_joint_3d_list,
                     image_file_path_list)


def main():
    if os.name == 'nt':
        pkl_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_egopw_orig_hand_egopw_body\outputs.pkl'
    else:
        pkl_path = r'/CT/EgoMocap/work/EgocentricFullBody/work_dirs/egofullbody_test_egopw_orig_hand_egopw_body/outputs.pkl'

    visualize_result(pkl_path)

if __name__ == '__main__':
    main()