#  Copyright Jian Wang @ MPI-INF (c) 2023.

import pickle
from copy import deepcopy

import numpy as np
import open3d
import torch

from mmpose.core import keypoint_mpjpe
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
# from mmpose.datasets.datasets.egocentric.hand_joint_eval import hand_keypoints_3d_to_fisheye_camera_space
# from mmpose.models.ego_hand_pose_estimation.utils.human_models import MANO
# from mmpose.utils.geometry_utils.geometry_utils_torch import convert_points_to_homogeneous
from mmpose.utils.visualization.hand_skeleton import HandSkeleton
from mmpose.utils.visualization.skeleton import Skeleton


def convert_from_model_format_to_mo2cap2_format(joints, model_idxs, dst_idxs):
    mo2cap2_shape = list(joints.shape)
    mo2cap2_shape[1] = 15
    mo2cap2_joint_batch = np.empty(mo2cap2_shape)

    mo2cap2_joint_batch[:, dst_idxs] = joints[:, model_idxs]
    return mo2cap2_joint_batch

def convert_from_model_format_to_hand_format(joints, model_idxs, dst_idxs):
    hand_shape = list(joints.shape)
    hand_shape[1] = 21
    hand_joint_batch = np.empty(hand_shape)

    hand_joint_batch[:, dst_idxs] = joints[:, model_idxs]
    return hand_joint_batch


def main():
    body_skeleton = Skeleton(None)
    hand_skeleton = HandSkeleton('mano')

    dst_idxs, model_idxs = dset_to_body_model(dset='mo2cap2', model_type='renderpeople', use_face_contour=False)
    left_hand_idxs, left_hand_model_idxs = dset_to_body_model(dset='mano_left', model_type='renderpeople', use_face_contour=False)
    right_hand_idxs, right_hand_model_idxs = dset_to_body_model(dset='mano_right', model_type='renderpeople',
                                                     use_face_contour=False)
    pkl_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_b_256\results_smplx_ik.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    pred_joints_3d_list = []
    pred_left_hand_joint_3d_list = []
    pred_right_hand_joint_3d_list = []
    gt_joints_3d_list = []
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
            gt_joints_3d_list.append(gt_joints_3d_item)

    gt_joints_3d_list = np.array(gt_joints_3d_list)
    pred_joints_3d_list = np.array(pred_joints_3d_list)
    pred_left_hand_joint_3d_list = np.array(pred_left_hand_joint_3d_list)
    pred_right_hand_joint_3d_list = np.array(pred_right_hand_joint_3d_list)
    # convert from model format to mo2cap2 format
    original_gt_joint_3d_list = deepcopy(gt_joints_3d_list)
    gt_joints_3d_list = convert_from_model_format_to_mo2cap2_format(original_gt_joint_3d_list,
                                                                    model_idxs=model_idxs,
                                                                    dst_idxs=dst_idxs)
    gt_left_hand_joint_3d_list = convert_from_model_format_to_hand_format(original_gt_joint_3d_list,
                                                                             model_idxs=left_hand_model_idxs,
                                                                             dst_idxs=left_hand_idxs)
    gt_right_hand_joint_3d_list = convert_from_model_format_to_hand_format(original_gt_joint_3d_list,
                                                                              model_idxs=right_hand_model_idxs,
                                                                              dst_idxs=right_hand_idxs)
    mask = np.ones((gt_joints_3d_list.shape[0], gt_joints_3d_list.shape[1]), dtype=np.bool)
    mpjpe = keypoint_mpjpe(pred_joints_3d_list, gt_joints_3d_list, mask)

    print(mpjpe)
    pa_mpjpe = keypoint_mpjpe(pred_joints_3d_list, gt_joints_3d_list, mask, alignment='procrustes')
    print(pa_mpjpe)

    hand_mask = np.ones((gt_left_hand_joint_3d_list.shape[0], gt_left_hand_joint_3d_list.shape[1]), dtype=np.bool)
    left_hand_pa_mpjpe = keypoint_mpjpe(pred_left_hand_joint_3d_list,
                                        gt_left_hand_joint_3d_list, hand_mask, alignment='procrustes')
    print(left_hand_pa_mpjpe)
    right_hand_pa_mpjpe = keypoint_mpjpe(pred_right_hand_joint_3d_list,
                                            gt_right_hand_joint_3d_list, hand_mask, alignment='procrustes')
    print(right_hand_pa_mpjpe)

    # transform the hand root joint to the wrist joint of body pose
    assert len(pred_left_hand_joint_3d_list) == len(pred_right_hand_joint_3d_list) == len(pred_joints_3d_list)
    for i in range(len(pred_joints_3d_list)):
        pred_left_hand = pred_left_hand_joint_3d_list[i]
        pred_right_hand = pred_right_hand_joint_3d_list[i]
        pred_body_pose = pred_joints_3d_list[i]
        pred_right_hand += pred_body_pose[3] - pred_right_hand[0]
        pred_left_hand += pred_body_pose[6] - pred_left_hand[0]

        body_mesh = body_skeleton.joints_2_mesh(pred_body_pose)
        left_hand_mesh = hand_skeleton.joints_2_mesh(pred_left_hand)
        right_hand_mesh = hand_skeleton.joints_2_mesh(pred_right_hand)
        open3d.visualization.draw_geometries([body_mesh, left_hand_mesh, right_hand_mesh])


if __name__ == '__main__':
    main()
