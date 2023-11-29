#  Copyright Jian Wang @ MPI-INF (c) 2023.
import os
import pickle

import numpy as np
import open3d
import smplx
import torch
from tqdm import tqdm

import mmpose.models.ego_full_body.smplx_ik.geometry_utils as gu
from mmpose.utils.visualization.hand_skeleton import HandSkeleton
from mmpose.utils.visualization.skeleton import Skeleton


def get_kinematic_map(smplx_model, dst_idx):
    cur = dst_idx
    kine_map = dict()
    while cur >= 0:
        parent = int(smplx_model.parents[cur])
        if cur != dst_idx:  # skip the dst_idx itself
            kine_map[parent] = cur
        cur = parent
    return kine_map


def __transfer_rot(body_pose_rotmat, part_rotmat, kinematic_map, transfer_type):
    rotmat = body_pose_rotmat[0]
    parent_id = 0
    while parent_id in kinematic_map:
        child_id = kinematic_map[parent_id]
        local_rotmat = body_pose_rotmat[child_id]
        rotmat = torch.matmul(rotmat, local_rotmat)
        parent_id = child_id

    if transfer_type == 'g2l':
        part_rot_new = torch.matmul(rotmat.T, part_rotmat)
    else:
        assert transfer_type == 'l2g'
        part_rot_new = torch.matmul(rotmat, part_rotmat)

    return part_rot_new


def transfer_rotation(
        smplx_model, body_pose, part_rot, part_idx,
        transfer_type="g2l", result_format="rotmat"):
    assert transfer_type in ["g2l", "l2g"]
    assert result_format in ['rotmat', 'aa']

    assert type(body_pose) == type(part_rot)
    return_np = False

    if isinstance(body_pose, np.ndarray):
        body_pose = torch.from_numpy(body_pose)
        return_np = True

    if isinstance(part_rot, np.ndarray):
        part_rot = torch.from_numpy(part_rot)
        return_np = True

    if body_pose.dim() == 2:
        # aa
        assert body_pose.size(0) == 1 and body_pose.size(1) in [66, 72]
        body_pose_rotmat = gu.angle_axis_to_rotation_matrix(body_pose.view(22, 3)).clone()
    else:
        # rotmat
        assert body_pose.dim() == 4
        assert body_pose.size(0) == 1 and body_pose.size(1) in [22, 24]
        assert body_pose.size(2) == 3 and body_pose.size(3) == 3
        body_pose_rotmat = body_pose[0].clone()

    if part_rot.dim() == 2:
        # aa
        assert part_rot.size(0) == 1 and part_rot.size(1) == 3
        part_rotmat = gu.angle_axis_to_rotation_matrix(part_rot)[0, :3, :3].clone()
    else:
        # rotmat
        assert part_rot.dim() == 3
        assert part_rot.size(0) == 1 and part_rot.size(1) == 3 and part_rot.size(2) == 3
        part_rotmat = part_rot[0, :3, :3].clone()

    kinematic_map = get_kinematic_map(smplx_model, part_idx)
    part_rot_trans = __transfer_rot(
        body_pose_rotmat, part_rotmat, kinematic_map, transfer_type)

    if result_format == 'rotmat':
        return_value = part_rot_trans
    else:
        part_rot_aa = gu.rotation_matrix_to_angle_axis(part_rot_trans)
        return_value = part_rot_aa
    if return_np:
        return_value = return_value.numpy()
    return return_value


def transfer_rotation_debug(
        smplx_model, body_pose, part_rot, part_idx,
        transfer_type="g2l", result_format="rotmat", hand_transform=None):
    assert transfer_type in ["g2l", "l2g"]
    assert result_format in ['rotmat', 'aa']

    assert type(body_pose) == type(part_rot)
    return_np = False

    if isinstance(body_pose, np.ndarray):
        body_pose = torch.from_numpy(body_pose)
        return_np = True

    if isinstance(part_rot, np.ndarray):
        part_rot = torch.from_numpy(part_rot)
        return_np = True

    if body_pose.dim() == 2:
        # aa
        assert body_pose.size(0) == 1 and body_pose.size(1) in [66, 72]
        body_pose_rotmat = gu.angle_axis_to_rotation_matrix(body_pose.view(22, 3)).clone()
    else:
        # rotmat
        assert body_pose.dim() == 4
        assert body_pose.size(0) == 1 and body_pose.size(1) in [22, 24]
        assert body_pose.size(2) == 3 and body_pose.size(3) == 3
        body_pose_rotmat = body_pose[0].clone()

    if part_rot.dim() == 2:
        # aa
        assert part_rot.size(0) == 1 and part_rot.size(1) == 3
        part_rotmat = gu.angle_axis_to_rotation_matrix(part_rot)[0, :3, :3].clone()
    else:
        # rotmat
        assert part_rot.dim() == 3
        assert part_rot.size(0) == 1 and part_rot.size(1) == 3 and part_rot.size(2) == 3
        part_rotmat = part_rot[0, :3, :3].clone()

    part_rotmat = (hand_transform.cpu().float()) @ part_rotmat

    kinematic_map = get_kinematic_map(smplx_model, part_idx)
    part_rot_trans = __transfer_rot(
        body_pose_rotmat, part_rotmat, kinematic_map, transfer_type)

    if result_format == 'rotmat':
        return_value = part_rot_trans
    else:
        part_rot_aa = gu.rotation_matrix_to_angle_axis(part_rot_trans)
        return_value = part_rot_aa
    if return_np:
        return_value = return_value.numpy()
    return return_value


def integration_body_hand(smplx_model, body_info, left_hand_image_rot_mat, right_hand_image_rot_mat):
    left_hand_pose = body_info['left_hand_pose']
    left_hand_global_orient = body_info['left_hand_rot']
    right_hand_pose = body_info['right_hand_pose']
    right_hand_global_orient = body_info['right_hand_rot']

    # copy and paste
    pred_betas = torch.as_tensor(body_info['betas'])
    pred_global_orient = torch.as_tensor(body_info['global_orient'])
    pred_pose = torch.as_tensor(body_info['body_pose'])
    pred_transl = torch.as_tensor(body_info['transl'])
    # combine body pose and global orient
    pred_pose = torch.concatenate([pred_global_orient, pred_pose], dim=-1)
    pred_pose = pred_pose.view((-1, 22, 3))

    pred_rotmat = gu.angle_axis_to_rotation_matrix(pred_pose)

    # integrate right hand pose
    right_hand_local_orient = transfer_rotation_debug(
        smplx_model, pred_rotmat, right_hand_global_orient, 21, hand_transform=right_hand_image_rot_mat)
    # right_hand_image_rot_mat = right_hand_image_rot_mat.to(right_hand_local_orient.device)
    # pred_rotmat[0, 21] = right_hand_local_orient @ right_hand_image_rot_mat.T.to(right_hand_local_orient.device)
    # pred_rotmat[0, 21] = (right_hand_image_rot_mat @ right_hand_local_orient.T).T
    pred_rotmat[0, 21] = right_hand_local_orient

    # integrate left hand pose
    left_hand_local_orient = transfer_rotation_debug(
        smplx_model, pred_rotmat, left_hand_global_orient, 20, hand_transform=left_hand_image_rot_mat)
    pred_rotmat[0, 20] = left_hand_local_orient
    # left_hand_local_orient = transfer_rotation(
    #     smplx_model, pred_rotmat, left_hand_global_orient, 20)
    # pred_rotmat[0, 20] = left_hand_image_rot_mat.to(left_hand_local_orient.device) @ left_hand_local_orient

    pred_aa = gu.rotation_matrix_to_angle_axis(pred_rotmat)
    pred_aa = pred_aa.reshape(pred_aa.shape[0], 66)

    param_output = {
        'betas': pred_betas,
        'body_pose': pred_aa[:, 3:],
        'global_orient': pred_aa[:, :3],
        'right_hand_pose': right_hand_pose,
        'left_hand_pose': left_hand_pose,
        'transl': pred_transl
    }
    return param_output


def combine_body_hand_from_ik_results(smplx_model_path, ik_result_path, save_path=None, vis=False):
    smplx_model = smplx.create(smplx_model_path, model_type='smplx', gender='NEUTRAL', use_pca=False, num_betas=16,
                               batch_size=256)
    with open(ik_result_path, 'rb') as f:
        full_body_results_with_ik = pickle.load(f)
    for i, pred in tqdm(enumerate(full_body_results_with_ik)):
        pred_body_joints_3d = pred['body_pose_results']['keypoints_pred']
        ik_results = pred['smplx_ik']
        left_hands_preds = pred['left_hands_preds']
        right_hands_preds = pred['right_hands_preds']
        left_hand_joints = left_hands_preds['joint_cam_transformed']
        right_hand_joints = right_hands_preds['joint_cam_transformed']
        left_hand_transform = pred['left_hand_transform'][:, :3, :3].float()
        right_hand_transform = pred['right_hand_transform'][:, :3, :3].float()
        # print(left_hand_transform.shape)

        left_hand_mano = left_hands_preds['mano_pose']
        left_hand_mano = left_hand_mano.reshape(len(left_hand_mano), -1, 3)
        left_hand_mano[:, :, 1: 3] *= -1
        left_hand_mano = left_hand_mano.reshape(len(left_hand_mano), -1)
        left_hand_rot = left_hand_mano[:, :3]

        smplx_param = {
            'betas': ik_results['betas'],
            'body_pose': ik_results['pose_body'],
            'global_orient': ik_results['root_orient'],
            'transl': ik_results['trans'],
            'left_hand_pose': left_hand_mano[:, 3:],
            'right_hand_pose': right_hands_preds['mano_pose'][:, 3:],
            'left_hand_rot': left_hand_rot,
            'right_hand_rot': right_hands_preds['mano_pose'][:, :3]
        }
        smplx_param_output_list = []
        for j in range(smplx_param['betas'].shape[0]):
            smplx_param_item = {
                'betas': smplx_param['betas'][j].unsqueeze(0).cpu(),
                'body_pose': smplx_param['body_pose'][j].unsqueeze(0).cpu(),
                'global_orient': smplx_param['global_orient'][j].unsqueeze(0).cpu(),
                'transl': smplx_param['transl'][j].unsqueeze(0).cpu(),
                'left_hand_pose': smplx_param['left_hand_pose'][j].unsqueeze(0).cpu(),
                'right_hand_pose': smplx_param['right_hand_pose'][j].unsqueeze(0).cpu(),
                'left_hand_rot': smplx_param['left_hand_rot'][j].unsqueeze(0).cpu(),
                'right_hand_rot': smplx_param['right_hand_rot'][j].unsqueeze(0).cpu()
            }

            smplx_param_output = integration_body_hand(smplx_model, smplx_param_item,
                                                       left_hand_transform[j], right_hand_transform[j])
            smplx_param_output_list.append(smplx_param_output)

        smplx_param_output = {
            'betas': torch.cat([item['betas'] for item in smplx_param_output_list], dim=0),
            'body_pose': torch.cat([item['body_pose'] for item in smplx_param_output_list], dim=0),
            'global_orient': torch.cat([item['global_orient'] for item in smplx_param_output_list], dim=0),
            'transl': torch.cat([item['transl'] for item in smplx_param_output_list], dim=0),
            'left_hand_pose': torch.cat([item['left_hand_pose'] for item in smplx_param_output_list], dim=0),
            'right_hand_pose': torch.cat([item['right_hand_pose'] for item in smplx_param_output_list], dim=0),
        }

        full_body_results_with_ik[i]['smplx_param'] = smplx_param_output

        if vis:
            # visualize smplx model
            smplx_output = smplx_model(**smplx_param_output)
            vertices = smplx_output.vertices.detach().cpu().numpy()
            joints = smplx_output.joints.detach().cpu().numpy()
            faces = smplx_model.faces
            for j in range(0, len(vertices), 100):
                pred_left_hand = left_hand_joints[j]
                pred_right_hand = right_hand_joints[j]
                pred_body_pose = pred_body_joints_3d[j]
                pred_right_hand += pred_body_pose[3] - pred_right_hand[0].cpu().numpy()
                pred_left_hand += pred_body_pose[6] - pred_left_hand[0].cpu().numpy()

                body_pose_mesh = body_skeleton.joints_2_mesh(pred_body_pose)
                left_hand_mesh = hand_skeleton.joints_2_mesh(pred_left_hand)
                right_hand_mesh = hand_skeleton.joints_2_mesh(pred_right_hand)

                body_smpl_mesh = open3d.geometry.TriangleMesh()
                body_smpl_mesh.vertices = open3d.utility.Vector3dVector(vertices[j])
                body_smpl_mesh.triangles = open3d.utility.Vector3iVector(faces)
                body_smpl_mesh.compute_vertex_normals()
                open3d.visualization.draw_geometries([body_smpl_mesh, body_pose_mesh, left_hand_mesh, right_hand_mesh])
    # save full body results!
    if save_path is not None:
        print('save')
        with open(save_path, 'wb') as f:
            pickle.dump(full_body_results_with_ik, f)

if __name__ == '__main__':
    body_skeleton = Skeleton(None)
    hand_skeleton = HandSkeleton('mano')
    if os.name == 'nt':
        ik_result_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_b_256\results_smplx_ik.pkl'
        smplx_model_path = r'Z:\EgoMocap\work\EgocentricFullBody\3rdparty\human_body_prior\support_data\downloads\models_lockedhead\smplx\SMPLX_NEUTRAL.npz'
        save_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_b_256\results_smplx_ik_combined.pkl'
        vis = True
    else:
        ik_result_path = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/egofullbody_test_b_256/results_smplx_ik.pkl'
        smplx_model_path = '/CT/EgoMocap/work/EgocentricFullBody/3rdparty/human_body_prior/support_data/downloads/models_lockedhead/smplx/SMPLX_NEUTRAL.npz'
        save_path = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/egofullbody_test_b_256/results_smplx_ik_combined.pkl'
        vis = False

    combine_body_hand_from_ik_results(smplx_model_path, ik_result_path, save_path, vis)
