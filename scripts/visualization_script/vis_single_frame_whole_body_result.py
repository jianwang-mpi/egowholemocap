#  Copyright Jian Wang @ MPI-INF (c) 2023.

import pickle
from copy import deepcopy

import numpy as np
import open3d
import torch
from mmpose.utils.visualization.draw import draw_skeleton_with_chain
from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
from mmpose.data.keypoints_mapping.mano import mano_skeleton

def main(pkl_path, image_id):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    pred_joints_3d_list = []
    pred_left_hand_joint_3d_list = []
    pred_right_hand_joint_3d_list = []
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

    pred_joints_3d_list = np.array(pred_joints_3d_list)
    pred_left_hand_joint_3d_list = np.array(pred_left_hand_joint_3d_list)
    pred_right_hand_joint_3d_list = np.array(pred_right_hand_joint_3d_list)

    # transform the hand root joint to the wrist joint of body pose
    assert len(pred_left_hand_joint_3d_list) == len(pred_right_hand_joint_3d_list) == len(pred_joints_3d_list)

    pred_left_hand = pred_left_hand_joint_3d_list[image_id]
    pred_right_hand = pred_right_hand_joint_3d_list[image_id]
    pred_body_pose = pred_joints_3d_list[image_id]
    pred_right_hand += pred_body_pose[3] - pred_right_hand[0]
    pred_left_hand += pred_body_pose[6] - pred_left_hand[0]

    body_mesh = draw_skeleton_with_chain(pred_body_pose, mo2cap2_chain)
    left_hand_mesh = draw_skeleton_with_chain(pred_left_hand, mano_skeleton, keypoint_radius=0.01,
                                                      line_radius=0.0025)
    right_hand_mesh = draw_skeleton_with_chain(pred_right_hand, mano_skeleton, keypoint_radius=0.01,
                                                      line_radius=0.0025)
    open3d.visualization.draw_geometries([body_mesh, left_hand_mesh, right_hand_mesh])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='visualize single frame whole body result')
    parser.add_argument('--pred_path', type=str, required=True, help='prediction output pkl file path')
    parser.add_argument('--image_id', type=int, required=True, help='the image id to visualize')
    args = parser.parse_args()

    main(args.pred_path, args.image_id)
