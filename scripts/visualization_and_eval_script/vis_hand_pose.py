#  Copyright Jian Wang @ MPI-INF (c) 2023.

import pickle
from copy import deepcopy
import mmpose.models.ego_full_body.smplx_ik.geometry_utils as gu
import numpy as np
import open3d
import smplx
import torch

from mmpose.core import keypoint_mpjpe
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
from mmpose.models.ego_hand_pose_estimation.utils.human_models import HumanModels
from mmpose.utils.visualization.draw import draw_keypoints_3d
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

def get_hand_joint_mano(mano_model, mano_params, mano_joint_regressor, vis=False):
    mano_output = mano_model(**mano_params)
    vertices = mano_output.vertices
    batch_size = len(vertices)
    joint_cam = torch.bmm(
        torch.from_numpy(mano_joint_regressor)[None, :, :].repeat(batch_size, 1, 1),
        vertices)
    vertices_np = vertices.cpu().numpy()
    faces = mano_model.faces
    if vis:
        for i in range(0, len(vertices_np), 100):
            body_smpl_mesh = open3d.geometry.TriangleMesh()
            body_smpl_mesh.vertices = open3d.utility.Vector3dVector(vertices_np[i])
            body_smpl_mesh.triangles = open3d.utility.Vector3iVector(faces)
            body_smpl_mesh.compute_vertex_normals()

            hand_keypoints = joint_cam[i].cpu().numpy()
            hand_keypoints_mesh = draw_keypoints_3d(hand_keypoints, radius=0.008)
            open3d.visualization.draw_geometries([body_smpl_mesh, hand_keypoints_mesh])
    return joint_cam
def transform_hand_pose(hand_pose, transform):
    # hand pose shape: (batch_size, 21, 3)
    # transform shape: (batch_size, 4, 4)
    # transform hand pose to camera space
    transform = torch.asarray(transform).float().to(hand_pose.device)
    transformed_hand_pose = torch.bmm(transform[:, :3, :3], hand_pose.permute(0, 2, 1)).permute(0, 2, 1)
    # the translation part of the transformer is not used
    # transformed_hand_pose = transformed_hand_pose + transform[:, :3, 3:]
    return transformed_hand_pose

def main():
    body_skeleton = Skeleton(None)
    hand_skeleton = HandSkeleton('mano')
    right_mano_model = smplx.create('Z:\EgoMocap\work\EgocentricFullBody\human_models\mano\MANO_RIGHT.pkl',
                                    model_type='mano', is_rhand=True, use_pca=False, flat_hand_mean=False,
                                    batch_size=256)
    left_mano_model = smplx.create('Z:\EgoMocap\work\EgocentricFullBody\human_models\mano\MANO_LEFT.pkl',
                                    model_type='mano', is_rhand=False, use_pca=False, flat_hand_mean=False,
                                    batch_size=256)
    human_models = HumanModels()
    left_mano_model = human_models.mano.layer['left']
    right_mano_model = human_models.mano.layer['right']
    mano_joint_regressor = human_models.mano.joint_regressor


    dst_idxs, model_idxs = dset_to_body_model(dset='mo2cap2', model_type='renderpeople', use_face_contour=False)
    mo2cap2_smplx_dst_idxs, smplx_model_idxs = dset_to_body_model(dset='mo2cap2', model_type='smplx', use_face_contour=False)
    left_hand_idxs, left_hand_model_idxs = dset_to_body_model(dset='mano_left', model_type='renderpeople', use_face_contour=False)
    right_hand_idxs, right_hand_model_idxs = dset_to_body_model(dset='mano_right', model_type='renderpeople',
                                                     use_face_contour=False)
    # pkl_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_new_hand\results.pkl'
    pkl_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_b_256\results.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    pred_joints_3d_list = []
    pred_left_hand_joint_3d_list = []
    pred_right_hand_joint_3d_list = []
    pred_left_mano_joints_list = []
    pred_right_mano_joints_list = []
    gt_joints_3d_list = []
    for _, pred in enumerate(data):
        pred_joints_3d_item = pred['body_pose_results']['keypoints_pred']
        pred_left_hand_joint_item = pred['left_hands_preds']['joint_cam_transformed']
        pred_right_hand_joint_item = pred['right_hands_preds']['joint_cam_transformed']
        left_hand_transform = pred['left_hand_transform']
        right_hand_transform = pred['right_hand_transform']
        # pred_right_mano_pose = pred['right_hands_preds']['mano_pose']
        # right_mano_input = {
        #     'global_orient': pred_right_mano_pose[:, :3],
        #     'hand_pose': pred_right_mano_pose[:, 3:],
        # }
        # vis_hand_mano(right_mano_model, right_mano_input)

        pred_left_mano_pose = pred['left_hands_preds']['mano_pose']
        pred_left_mano_pose = pred_left_mano_pose.reshape(len(pred_left_mano_pose), -1, 3)
        pred_left_mano_pose[:, :, 1: 3] *= -1
        pred_left_mano_pose = pred_left_mano_pose.reshape(len(pred_left_mano_pose), -1)
        pred_left_mano_shape = pred['left_hands_preds']['mano_shape']

        pref_left_mano_global_orient = pred_left_mano_pose[:, :3]
        part_rotmat = gu.angle_axis_to_rotation_matrix(pref_left_mano_global_orient)[:, :3, :3].clone()
        # part_rotmat = torch.bmm(part_rotmat, left_hand_transform[:, :3, :3].cpu().float().permute(0, 2, 1))
        pref_left_mano_global_orient = gu.rotation_matrix_to_angle_axis(part_rotmat).clone()

        left_mano_input = {
            'global_orient': pref_left_mano_global_orient,
            'hand_pose': pred_left_mano_pose[:, 3:],
            'betas': pred_left_mano_shape
        }
        left_hand_mano_joint = get_hand_joint_mano(left_mano_model, left_mano_input, mano_joint_regressor)
        left_hand_mano_joint = transform_hand_pose(left_hand_mano_joint, left_hand_transform)

        pred_right_mano_pose = pred['right_hands_preds']['mano_pose']
        pred_right_mano_shape = pred['right_hands_preds']['mano_shape']
        pred_right_mano_global_orient = pred_right_mano_pose[:, :3]
        part_rotmat = gu.angle_axis_to_rotation_matrix(pred_right_mano_global_orient).clone()
        part_rotmat = torch.bmm(right_hand_transform[:, :3, :3].cpu().float(), part_rotmat)
        pred_right_mano_global_orient = gu.rotation_matrix_to_angle_axis(part_rotmat).clone()

        right_mano_input = {
            'global_orient': pred_right_mano_global_orient,
            'hand_pose': pred_right_mano_pose[:, 3:],
            'betas': pred_right_mano_shape
        }
        right_hand_mano_joint = get_hand_joint_mano(right_mano_model, right_mano_input, mano_joint_regressor)
        # right_hand_mano_joint = transform_hand_pose(right_hand_mano_joint, right_hand_transform)


        if torch.is_tensor(pred_joints_3d_item):
            pred_joints_3d_item = pred_joints_3d_item.cpu().numpy()
        if torch.is_tensor(pred_left_hand_joint_item):
            pred_left_hand_joint_item = pred_left_hand_joint_item.cpu().numpy()
            pred_right_hand_joint_item = pred_right_hand_joint_item.cpu().numpy()
            left_hand_mano_joint = left_hand_mano_joint.cpu().numpy()
            right_hand_mano_joint = right_hand_mano_joint.cpu().numpy()

        pred_joints_3d_list.extend(pred_joints_3d_item)
        pred_left_hand_joint_3d_list.extend(pred_left_hand_joint_item)
        pred_left_mano_joints_list.extend(left_hand_mano_joint)
        pred_right_mano_joints_list.extend(right_hand_mano_joint)
        pred_right_hand_joint_3d_list.extend(pred_right_hand_joint_item)
        img_meta_list = pred['img_metas']
        for img_meta_item in img_meta_list:
            gt_joints_3d_item = img_meta_item['keypoints_3d']
            gt_joints_3d_list.append(gt_joints_3d_item)

    gt_joints_3d_list = np.array(gt_joints_3d_list)
    pred_joints_3d_list = np.array(pred_joints_3d_list)
    pred_left_hand_joint_3d_list = np.array(pred_left_hand_joint_3d_list)
    pred_left_mano_joints_list = np.array(pred_left_mano_joints_list)
    pred_right_mano_joints_list = np.array(pred_right_mano_joints_list)
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
    mask = np.ones((gt_joints_3d_list.shape[0], gt_joints_3d_list.shape[1]), dtype=bool)
    mpjpe = keypoint_mpjpe(pred_joints_3d_list, gt_joints_3d_list, mask)

    print(mpjpe)
    pa_mpjpe = keypoint_mpjpe(pred_joints_3d_list, gt_joints_3d_list, mask, alignment='procrustes')
    print(pa_mpjpe)


    hand_mask = np.ones((gt_left_hand_joint_3d_list.shape[0], gt_left_hand_joint_3d_list.shape[1]), dtype=bool)
    left_hand_pa_mpjpe = keypoint_mpjpe(pred_left_hand_joint_3d_list,
                                        gt_left_hand_joint_3d_list, hand_mask, alignment='procrustes')
    print(left_hand_pa_mpjpe)
    right_hand_pa_mpjpe = keypoint_mpjpe(pred_right_hand_joint_3d_list,
                                            gt_right_hand_joint_3d_list, hand_mask, alignment='procrustes')
    print(right_hand_pa_mpjpe)

    # transform the hand root joint to the wrist joint of body pose
    assert len(pred_left_hand_joint_3d_list) == len(pred_right_hand_joint_3d_list) == len(pred_joints_3d_list)
    for i in range(0, len(pred_joints_3d_list), 100):
        pred_left_hand = pred_left_hand_joint_3d_list[i]
        gt_left_hand = gt_left_hand_joint_3d_list[i]
        pred_right_hand = pred_right_hand_joint_3d_list[i]
        mano_left_hand = pred_left_mano_joints_list[i]
        mano_right_hand = pred_right_mano_joints_list[i]
        gt_right_hand = gt_right_hand_joint_3d_list[i]
        pred_body_pose = pred_joints_3d_list[i]
        pred_right_hand += pred_body_pose[3] - pred_right_hand[0]
        gt_right_hand += pred_body_pose[3] - gt_right_hand[0]
        pred_left_hand += pred_body_pose[6] - pred_left_hand[0]
        gt_left_hand += pred_body_pose[6] - gt_left_hand[0]
        mano_left_hand += pred_body_pose[6] - mano_left_hand[0]
        mano_right_hand += pred_body_pose[3] - mano_right_hand[0]

        body_mesh = body_skeleton.joints_2_mesh(pred_body_pose)
        left_hand_mesh = hand_skeleton.joints_2_mesh(pred_left_hand)
        left_hand_mano_mesh = hand_skeleton.joints_2_mesh(mano_left_hand, joint_color=(1, 0, 1), bone_color=(1, 0, 1))
        right_hand_mano_mesh = hand_skeleton.joints_2_mesh(mano_right_hand, joint_color=(1, 0, 1), bone_color=(1, 0, 1))
        left_hand_gt_mesh = hand_skeleton.joints_2_mesh(gt_left_hand, joint_color=(1, 0, 0), bone_color=(1, 0, 0))
        right_hand_mesh = hand_skeleton.joints_2_mesh(pred_right_hand)
        right_hand_gt_mesh = hand_skeleton.joints_2_mesh(gt_right_hand, joint_color=(1, 0, 0), bone_color=(1, 0, 0))
        open3d.visualization.draw_geometries([body_mesh, left_hand_mesh, right_hand_mesh, left_hand_mano_mesh, right_hand_mano_mesh,
                                               left_hand_gt_mesh, right_hand_gt_mesh])


if __name__ == '__main__':
    main()
