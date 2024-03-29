import json
import os
import pickle

import numpy as np
import open3d
import torch
from mmpose.models.diffusion_mdm.data_loaders.humanml.common.quaternion import qrot_np, qbetween_np, qinv_np, qmul_np, euler_to_quaternion
from mmpose.core import keypoint_mpjpe
from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import Collect
from mmpose.models.diffusion_mdm.data_loaders.humanml.scripts.motion_process import (
    recover_from_ric,
)
from mmpose.datasets.pipelines import (Collect, AlignAllGlobalSMPLXJointsWithGlobalInfo,
                                       PreProcessHandMotion, SplitGlobalSMPLXJoints,
                                       PreProcessMo2Cap2BodyMotion)
if os.name == "nt":
    network_pred_seq_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_b_256\results.pkl'
    mean_std_path = r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\egobody\global_aligned_mean_std.pkl'
    diffusion_model_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\diffusion_full_body_train_uncond_eb_stu_rp\epoch_12.pth'
    image_data_dir = r'X:\ScanNet\work\egocentric_view\25082022'
    diffusion_result_save_global_dir = r"Z:/EgoMocap/work/EgocentricFullBody/vis_results/egofullbody_diffusion_results_global"
    diffusion_result_save_dir = r"Z:/EgoMocap/work/EgocentricFullBody/vis_results/egofullbody_diffusion_results"

else:
    network_pred_seq_path = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/egofullbody_test_b_256/results.pkl'
    mean_std_path = '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/egobody/global_aligned_mean_std.pkl'
    image_data_dir = r'/CT/ScanNet/work/egocentric_view/25082022'
    diffusion_model_path = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/diffusion_full_body_train_uncond_eb_stu_rp/epoch_12.pth'
    diffusion_result_save_global_dir = "/CT/EgoMocap/work/EgocentricFullBody/vis_results/egofullbody_diffusion_results_global_info"
    diffusion_result_save_dir = "/CT/EgoMocap/work/EgocentricFullBody/vis_results/egofullbody_diffusion_results"


def get_fullbody_motion_test_dataset():
    seq_len = 196
    normalize = True
    pipeline = [
        AlignAllGlobalSMPLXJointsWithGlobalInfo(use_default_floor_height=True),
        SplitGlobalSMPLXJoints(smplx_joint_name='aligned_smplx_joints'),
        PreProcessHandMotion(normalize=normalize,
                             mean_std_path=mean_std_path),
        PreProcessMo2Cap2BodyMotion(normalize=normalize,
                                    mean_std_path=mean_std_path),
        Collect(keys=['aligned_smplx_joints',
                      'mo2cap2_body_features', 'left_hand_features', 'right_hand_features',
                      'processed_left_hand_keypoints_3d', 'processed_right_hand_keypoints_3d'],
                meta_keys=['gt_joints_3d', 'ego_camera_pose', 'root_trans_init_xz', 'root_rot_quat_init',
                           'root_trans_xz', 'root_rot_quat', 'local_root_velocity', 'local_joints_velocity',
                           'local_root_rotation_velocity', 'local_root_rotation_velocity_y',
                           'global_smplx_joints', 'ext_id', 'seq_name'])
    ]

    dataset_cfg = dict(
        type='FullBodyMotionTestDataset',
        data_pkl_path=network_pred_seq_path,
        seq_len=seq_len,
        skip_frames=seq_len,
        pipeline=pipeline,
        split_sequence=True,
        test_mode=True,
    )

    fullbody_motion_test_dataset = build_dataset(dataset_cfg)
    print(f"length of dataset is: {len(fullbody_motion_test_dataset)}")
    return fullbody_motion_test_dataset


def get_data(dataset, data_id, vis=False):
    data_i = dataset[data_id]

    if vis:
        coord = open3d.geometry.TriangleMesh.create_coordinate_frame()

    return data_i


def run_diffusion(data_id):
    full_body_motion_dataset = get_fullbody_motion_test_dataset()
    data_i = get_data(full_body_motion_dataset, data_id, vis=False)

    data_i = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v for k, v in data_i.items() }

    from mmpose.models.diffusion_hands.refine_mo2cap2_hands import RefineEdgeDiffusionHands
    full_body_pose_diffusion_refiner = RefineEdgeDiffusionHands(seq_len=196).cuda()
    full_body_pose_diffusion_refiner.eval()
    from mmcv.runner import load_checkpoint
    load_checkpoint(full_body_pose_diffusion_refiner, diffusion_model_path, map_location='cpu', strict=True)

    with torch.no_grad():
        diffusion_results = full_body_pose_diffusion_refiner(**data_i)

    print("diffusion_results: ", diffusion_results.keys())
    return diffusion_results


def get_global_root_rot_trans(local_root_velocity, local_root_rot_velocity):
    init_root_quat = np.array([1, 0, 0, 0])
    init_root_trans = np.array([0, 0, 0])
    # recover the root rotation for each frame
    seq_len = len(local_root_rot_velocity)
    root_quat = np.zeros((seq_len + 1, 4))
    root_trans = np.zeros((seq_len + 1, 3))
    root_quat[0] = init_root_quat
    root_trans[0] = init_root_trans
    for i in range(seq_len):
        root_quat[i + 1] = qmul_np(local_root_rot_velocity[i], root_quat[i])
        root_trans[i + 1] = qrot_np(qinv_np(root_quat[i]), local_root_velocity[i]) + root_trans[i]
    return -root_trans, root_quat

def get_pred_gt_motion(data_pkl_path):
    # evaluate the error of diffusion results
    with open(data_pkl_path, 'rb') as f:
        diffusion_results = pickle.load(f)
    normalize = True

    full_body_motion_sequence_list = diffusion_results['sample'].cpu().numpy()
    mo2cap2_motion_sequence_list = full_body_motion_sequence_list[:, :, :15 * 3]
    left_hand_motion_sequence_list = full_body_motion_sequence_list[:, :, 15 * 3: 15 * 3 + 21 * 3]
    right_hand_motion_sequence_list = full_body_motion_sequence_list[:, :, 15 * 3 + 21 * 3:]

    full_body_input_motion_list = diffusion_results['full_body_features'].cpu().numpy()
    mo2cap2_input_motion_list = full_body_input_motion_list[:, :, :15 * 3]
    left_hand_input_motion_list = full_body_input_motion_list[:, :, 15 * 3: 15 * 3 + 21 * 3]
    right_hand_input_motion_list = full_body_input_motion_list[:, :, 15 * 3 + 21 * 3:]

    with open(mean_std_path, 'rb') as f:
        global_aligned_mean_std = pickle.load(f)

    if normalize:
        left_hand_mean = global_aligned_mean_std['left_hand_mean']
        left_hand_std = global_aligned_mean_std['left_hand_std']
        left_hand_motion_sequence_list = left_hand_motion_sequence_list * left_hand_std + left_hand_mean
        left_hand_input_motion_list = left_hand_input_motion_list * left_hand_std + left_hand_mean
        right_hand_mean = global_aligned_mean_std['right_hand_mean']
        right_hand_std = global_aligned_mean_std['right_hand_std']
        right_hand_motion_sequence_list = right_hand_motion_sequence_list * right_hand_std + right_hand_mean
        right_hand_input_motion_list = right_hand_input_motion_list * right_hand_std + right_hand_mean
        mo2cap2_body_mean = global_aligned_mean_std['mo2cap2_body_mean']
        mo2cap2_body_std = global_aligned_mean_std['mo2cap2_body_std']
        mo2cap2_motion_sequence_list = mo2cap2_motion_sequence_list * mo2cap2_body_std + mo2cap2_body_mean
        mo2cap2_input_motion_list = mo2cap2_input_motion_list * mo2cap2_body_std + mo2cap2_body_mean

    left_hand_pred_motion = left_hand_motion_sequence_list.reshape(-1, 21, 3)
    right_hand_pred_motion = right_hand_motion_sequence_list.reshape(-1, 21, 3)
    mo2cap2_pred_motion = mo2cap2_motion_sequence_list.reshape(-1, 15, 3)
    left_hand_pred_motion[:, 0] *= 0
    right_hand_pred_motion[:, 0] *= 0

    left_hand_input_motion = left_hand_input_motion_list.reshape(-1, 21, 3)
    left_hand_input_motion[:, 0] *= 0
    right_hand_input_motion = right_hand_input_motion_list.reshape(-1, 21, 3)
    right_hand_input_motion[:, 0] *= 0
    mo2cap2_input_motion = mo2cap2_input_motion_list.reshape(-1, 15, 3)

    # get the gt motion and camera params
    img_metas = diffusion_results['img_metas']
    ext_id_list = img_metas.data['ext_id']
    seq_name_list = img_metas.data['seq_name']
    gt_studio_joint_list = img_metas.data['gt_joints_3d']
    ego_camera_pose_list = img_metas.data['ego_camera_pose']
    # data_preprocess_trans_list_old = img_metas.data['root_trans_xz'].cpu().numpy()
    # data_preprocess_rot_quat_list = img_metas.data['root_rot_quat'].cpu().numpy()
    data_preprocess_rot_quat_init = img_metas.data['root_rot_quat_init']
    data_preprocess_root_trans_init_xz = img_metas.data['root_trans_init_xz']
    data_preprocess_local_root_velocity = img_metas.data['local_root_velocity'].cpu().numpy()
    data_preprocess_local_root_rotation_velocity = img_metas.data['local_root_rotation_velocity'].cpu().numpy()
    data_preprocess_local_root_rotation_velocity_y = img_metas.data['local_root_rotation_velocity_y'].cpu().numpy()

    local_root_rotation_velocity_y = np.zeros((data_preprocess_local_root_rotation_velocity.shape[0], 3))
    local_root_rotation_velocity_y[:, 1] = 1
    local_root_rotation_velocity_y = local_root_rotation_velocity_y * data_preprocess_local_root_rotation_velocity_y[:, None]
    local_root_rotation_velocity_y_recon = euler_to_quaternion(local_root_rotation_velocity_y, order='xyz')

    data_preprocess_trans_list, data_preprocess_rot_quat_list = get_global_root_rot_trans(
        data_preprocess_local_root_velocity, local_root_rotation_velocity_y_recon)
    global_smplx_joints = img_metas.data['global_smplx_joints']

    from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
    mano_left_idxs, renderpeople_idxs_mano_left = dset_to_body_model(dset='mano_left', model_type='renderpeople')
    mano_right_idxs, renderpeople_idxs_mano_right = dset_to_body_model(dset='mano_right', model_type='renderpeople')
    mo2cap2_idxs, renderpeople_idxs_mo2cap2 = dset_to_body_model(dset='mo2cap2', model_type='renderpeople')

    left_hand_gt_motion = np.zeros_like(left_hand_pred_motion)
    right_hand_gt_motion = np.zeros_like(right_hand_pred_motion)
    mo2cap2_gt_motion = np.zeros_like(mo2cap2_pred_motion)
    left_hand_gt_motion[:, mano_left_idxs] = gt_studio_joint_list[:, renderpeople_idxs_mano_left]
    right_hand_gt_motion[:, mano_right_idxs] = gt_studio_joint_list[:, renderpeople_idxs_mano_right]
    mo2cap2_gt_motion[:, mo2cap2_idxs] = gt_studio_joint_list[:, renderpeople_idxs_mo2cap2]

    mano_left_idxs, smplx_idxs_mano_left = dset_to_body_model(dset='mano_left', model_type='smplx')
    mano_right_idxs, smplx_idxs_mano_right = dset_to_body_model(dset='mano_right', model_type='smplx')
    mo2cap2_idxs, smplx_idxs_mo2cap2 = dset_to_body_model(dset='mo2cap2', model_type='smplx')

    left_hand_input_motion = np.zeros_like(left_hand_pred_motion)
    right_hand_input_motion = np.zeros_like(right_hand_pred_motion)
    mo2cap2_input_motion = np.zeros_like(mo2cap2_pred_motion)

    left_hand_input_motion[:, mano_left_idxs] = global_smplx_joints[:, smplx_idxs_mano_left]
    right_hand_input_motion[:, mano_right_idxs] = global_smplx_joints[:, smplx_idxs_mano_right]
    mo2cap2_input_motion[:, mo2cap2_idxs] = global_smplx_joints[:, smplx_idxs_mo2cap2]

    combined_input_motion = np.concatenate([mo2cap2_input_motion, left_hand_input_motion, right_hand_input_motion], axis=1)

    # recover the full body
    # rotate it back
    left_hand_root = mo2cap2_pred_motion[:, 6: 7]
    left_hand_pred_motion_on_body = left_hand_pred_motion + left_hand_root

    right_hand_root = mo2cap2_pred_motion[:, 3: 4]
    right_hand_pred_motion_on_body = right_hand_pred_motion + right_hand_root
    combined_motion = np.concatenate([mo2cap2_pred_motion,
                                      left_hand_pred_motion_on_body,
                                      right_hand_pred_motion_on_body], axis=1)

    data_preprocess_rot_quat_list = data_preprocess_rot_quat_list
    data_preprocess_rot_quat_inv_list = qinv_np(data_preprocess_rot_quat_list)
    data_preprocess_rot_quat_inv_list = np.repeat(data_preprocess_rot_quat_inv_list[:, None, :], combined_motion.shape[1], axis=1)
    combined_motion = qrot_np(data_preprocess_rot_quat_inv_list, combined_motion)
    if len(data_preprocess_trans_list.shape) == 2:
        data_preprocess_trans_list = data_preprocess_trans_list[:, None, :]
    combined_motion = combined_motion - data_preprocess_trans_list

    #rotate and translate with init rot and transl
    data_preprocess_rot_quat_init = data_preprocess_rot_quat_init.cpu().numpy()
    data_preprocess_rot_quat_init = qinv_np(data_preprocess_rot_quat_init)
    data_preprocess_rot_quat_init = np.ones(combined_motion.shape[:-1] + (4,)) * data_preprocess_rot_quat_init
    combined_motion = qrot_np(data_preprocess_rot_quat_init, combined_motion)
    combined_motion = combined_motion - data_preprocess_root_trans_init_xz.cpu().numpy()

    # visualize for debug
    # print('start visualization')
    # from mmpose.utils.visualization.draw import draw_skeleton_with_chain, draw_keypoints_3d
    # from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
    # from mmpose.data.keypoints_mapping.mano import mano_skeleton
    # # split combined motion
    # scene_mesh_path = r'X:\ScanNet\work\25-08-22\shot_bg\recon_bak2\ply\model.ply'
    # scene_mesh = open3d.io.read_triangle_mesh(scene_mesh_path)
    # scene_mesh = scene_mesh.scale(0.001, center=np.asarray([0, 0, 0]))
    # mo2cap2_pred_motion = combined_motion[:, :15]
    # left_hand_pred_motion = combined_motion[:, 15: 15 + 21]
    # right_hand_pred_motion = combined_motion[:, 15 + 21:]
    # for i in range(0, 196, 10):
    #     global_smplx_joints_mesh = draw_keypoints_3d(global_smplx_joints[i])
    #     mo2cap2_mesh = draw_skeleton_with_chain(mo2cap2_pred_motion[i], mo2cap2_chain)
    #     left_hand_mesh = draw_skeleton_with_chain(left_hand_pred_motion[i], mano_skeleton, keypoint_radius=0.01,
    #                                                       line_radius=0.0025)
    #     right_hand_mesh = draw_skeleton_with_chain(right_hand_pred_motion[i], mano_skeleton, keypoint_radius=0.01,
    #                                                       line_radius=0.0025)
    #     coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
    #     open3d.visualization.draw_geometries([mo2cap2_mesh, left_hand_mesh, right_hand_mesh, scene_mesh,
    #                                           global_smplx_joints_mesh])

    result = {'left_hand_pred_motion': left_hand_pred_motion,
              'left_hand_input_motion': left_hand_input_motion,
              'right_hand_pred_motion': right_hand_pred_motion,
              'right_hand_input_motion': right_hand_input_motion,
              'mo2cap2_pred_motion': mo2cap2_pred_motion,
              'mo2cap2_input_motion': mo2cap2_input_motion,
              'left_hand_gt_motion': left_hand_gt_motion,
              'right_hand_gt_motion': right_hand_gt_motion,
              'mo2cap2_gt_motion': mo2cap2_gt_motion,
              'ego_camera_pose_list': ego_camera_pose_list,
              'global_combined_input_motion': combined_input_motion,
              'global_combined_output_motion': combined_motion,
              'ext_id': ext_id_list,
              'seq_name': seq_name_list,}
    return result

def run_diffusion_and_visualize():
    data_id_input = 16
    for data_id_input in range(11, 65, 1):
        if data_id_input != 12:
            continue
        diff_results = run_diffusion(data_id_input)
        # save the results
        if not os.path.exists(diffusion_result_save_global_dir):
            os.makedirs(diffusion_result_save_global_dir)
        data_pkl_path = os.path.join(diffusion_result_save_global_dir, f"diffusion_results_{data_id_input}_ik.pkl")
        with open(data_pkl_path, 'wb') as f:
            pickle.dump(diff_results, f)

        # data_pkl_path = os.path.join(diffusion_result_save_global_dir, f"diffusion_results_{data_id_input}.pkl")
        result_data = get_pred_gt_motion(data_pkl_path)
        body_mask = np.ones((result_data['mo2cap2_pred_motion'].shape[0], 15)).astype(bool)
        left_hand_mask = np.ones((result_data['left_hand_pred_motion'].shape[0], 21)).astype(bool)
        right_hand_mask = np.ones((result_data['right_hand_pred_motion'].shape[0], 21)).astype(bool)
        body_motion_accuracy = keypoint_mpjpe(result_data['mo2cap2_pred_motion'], result_data['mo2cap2_gt_motion'],
                                              body_mask, alignment='procrustes')
        left_hand_motion_accuracy = keypoint_mpjpe(result_data['left_hand_pred_motion'], result_data['left_hand_gt_motion'],
                                                   left_hand_mask, alignment='procrustes')
        right_hand_motion_accuracy = keypoint_mpjpe(result_data['right_hand_pred_motion'], result_data['right_hand_gt_motion'],
                                                    right_hand_mask, alignment='procrustes')
        print('body mpjpe:', body_motion_accuracy)
        print('left hand mpjpe:', left_hand_motion_accuracy)
        print('right hand mpjpe:', right_hand_motion_accuracy)

        seq_name = result_data['seq_name'][0]
        seq_name = seq_name[seq_name.find('_') + 1:] if seq_name.startswith('new') else seq_name
        ext_id_start = result_data['ext_id'][0]
        seq_path = os.path.join(image_data_dir, seq_name)
        with open(os.path.join(seq_path, 'syn.json'), 'rb') as f:
            syn_data = json.load(f)
        ego_id_start = ext_id_start - syn_data['ext'] + syn_data['ego']
        ego_id_start = 'img_%06d' % ego_id_start
        # eval_motion_error(result_data)
        body_motion_accuracy = '%.2f' % (body_motion_accuracy * 1000)
        left_hand_motion_accuracy = '%.2f' % (left_hand_motion_accuracy * 1000)
        right_hand_motion_accuracy = '%.2f' % (right_hand_motion_accuracy * 1000)
        with open(os.path.join(diffusion_result_save_global_dir, f"ik_input_{data_id_input}_{seq_name}_{ego_id_start}"
                                                                 f"body_{body_motion_accuracy}"
                                                                 f"lh_{left_hand_motion_accuracy}"
                                                                 f"rh_{right_hand_motion_accuracy}.pkl"), 'wb') as f:
            pickle.dump(result_data, f)
    # visualize_results(data_id_input, result_data, save_dir=diffusion_result_save_dir, save_viewpoint=True, render=False)
    # visualize_input(data_id_input, result_data, save_dir=diffusion_result_save_dir, save_viewpoint=False, render=True)
    # visualize_results(data_id_input, result_data, save_dir=diffusion_result_save_global_dir, save_viewpoint=False, render=True)
    # data_id_input = 5
    # for data_id_input in range(0, 65, 2):
    #     run_diffusion(data_id_input)
    #     data_id_input = 48
    #     data_pkl_path = os.path.join(diffusion_result_save_dir, f"diffusion_results_{data_id_input}.pkl")
    #     result_data = get_pred_gt_motion(data_pkl_path)
    #     # visualize_results(data_id_input, result_data, save_dir=diffusion_result_save_dir, save_viewpoint=True, render=False)
    #     visualize_input(data_id_input, result_data, save_dir=diffusion_result_save_dir, save_viewpoint=False, render=True)
    #     visualize_results(data_id_input, result_data, save_dir=diffusion_result_save_dir, save_viewpoint=False, render=True)

if __name__ == '__main__':
    run_diffusion_and_visualize()
    # run_whole_sequence_and_evaluate()