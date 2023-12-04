# local diffusion model
import os
import pickle

import numpy as np
import open3d
import torch

from mmpose.core import keypoint_mpjpe
from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import Collect
from mmpose.models.diffusion_mdm.data_loaders.humanml.scripts.motion_process import (
    recover_from_ric,
)
from mmpose.utils.visualization.draw import draw_skeleton_with_chain, draw_keypoints_3d
from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
from mmpose.data.keypoints_mapping.mano import mano_skeleton
from mmpose.datasets.pipelines import (Collect, AlignAllGlobalSMPLXJointsWithInfo,
                                       PreProcessHandMotion, AlignGlobalSMPLXJoints, SplitGlobalSMPLXJoints,
                                       PreProcessMo2Cap2BodyMotion, ExtractConfidence)
if os.name == "nt":
    network_pred_seq_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_b_256\results.pkl'
    mean_std_path = r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\egobody\global_aligned_mean_std.pkl'
    diffusion_model_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\diffusion_full_body_train_uncond_eb_stu_rp\epoch_12.pth'
    diffusion_result_save_global_dir = r"Z:/EgoMocap/work/EgocentricFullBody/vis_results/with_uncertainty"

else:
    network_pred_seq_path = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/egofullbody_test_b_256/results.pkl'
    mean_std_path = '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/egobody/global_aligned_mean_std.pkl'
    diffusion_model_path = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/diffusion_full_body_train_uncond_eb_stu_rp/epoch_12.pth'
    diffusion_result_save_global_dir = "/CT/EgoMocap/work/EgocentricFullBody/vis_results/with_uncertainty"


def get_fullbody_motion_test_dataset():
    seq_len = 196
    normalize = True
    pipeline = [
        AlignAllGlobalSMPLXJointsWithInfo(use_default_floor_height=True),
        SplitGlobalSMPLXJoints(smplx_joint_name='aligned_smplx_joints'),
        PreProcessHandMotion(normalize=normalize,
                             mean_std_path=mean_std_path),
        PreProcessMo2Cap2BodyMotion(normalize=normalize,
                                    mean_std_path=mean_std_path),
        ExtractConfidence(confidence_name='human_body_confidence'),
        ExtractConfidence(confidence_name='left_hand_confidence'),
        ExtractConfidence(confidence_name='right_hand_confidence'),
        Collect(keys=['aligned_smplx_joints',
                      'mo2cap2_body_features', 'left_hand_features', 'right_hand_features',
                        'human_body_confidence', 'left_hand_confidence', 'right_hand_confidence',
                      'processed_left_hand_keypoints_3d', 'processed_right_hand_keypoints_3d'],
                meta_keys=['gt_joints_3d', 'ego_camera_pose', 'smplx_root_trans', 'root_quat',
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


def run_diffusion(data_id, save=True):
    full_body_motion_dataset = get_fullbody_motion_test_dataset()
    data_i = get_data(full_body_motion_dataset, data_id, vis=False)
    print('seq_name: {}'.format(data_i['img_metas'].data['seq_name']))
    print('ext id: {}'.format(data_i['img_metas'].data['ext_id']))

    data_i = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v for k, v in data_i.items() }

    from mmpose.models.diffusion_hands.refine_mo2cap2_hands_with_uncertainty import RefineEdgeDiffusionHandsUncertainty
    full_body_pose_diffusion_refiner = RefineEdgeDiffusionHandsUncertainty(seq_len=196).cuda()
    full_body_pose_diffusion_refiner.eval()
    from mmcv.runner import load_checkpoint
    load_checkpoint(full_body_pose_diffusion_refiner, diffusion_model_path, map_location='cpu', strict=True)

    with torch.no_grad():
        diffusion_results = full_body_pose_diffusion_refiner(**data_i)

    print("diffusion_results: ", diffusion_results.keys())

    # save the results
    if save:
        if not os.path.exists(diffusion_result_save_global_dir):
            os.makedirs(diffusion_result_save_global_dir)
        with open(os.path.join(diffusion_result_save_global_dir, f"diffusion_results_{data_id}.pkl"), 'wb') as f:
            pickle.dump(diffusion_results, f)
    return diffusion_results

def get_pred_gt_motion(data_pkl_path=None, diffusion_results_input=None):
    # evaluate the error of diffusion results
    if diffusion_results_input is None:
        with open(data_pkl_path, 'rb') as f:
            diffusion_results = pickle.load(f)
    else:
        diffusion_results = diffusion_results_input
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
    gt_studio_joint_list = img_metas.data['gt_joints_3d']
    ego_camera_pose_list = img_metas.data['ego_camera_pose']
    data_preprocess_trans_list = img_metas.data['smplx_root_trans']
    data_preprocess_rot_quat_list = img_metas.data['root_quat']
    global_smplx_joints = img_metas.data['global_smplx_joints']
    ext_id = img_metas.data['ext_id']
    seq_name = img_metas.data['seq_name']

    print('seq_name: ', seq_name)
    print('ext id: ', ext_id)

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
    from mmpose.models.diffusion_mdm.data_loaders.humanml.common.quaternion import qrot_np, qbetween_np, qinv_np
    data_preprocess_rot_quat_list = data_preprocess_rot_quat_list.cpu().numpy()
    data_preprocess_rot_quat_inv_list = qinv_np(data_preprocess_rot_quat_list)
    data_preprocess_rot_quat_inv_list = np.repeat(data_preprocess_rot_quat_inv_list[:, None, :], combined_motion.shape[1], axis=1)
    combined_motion = qrot_np(data_preprocess_rot_quat_inv_list, combined_motion)
    combined_motion = combined_motion - data_preprocess_trans_list.cpu().numpy()

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
              'seq_name': seq_name[0],
              'ext_id': ext_id}
    return result


def eval_smooth_motion_error(result_data):
    # only calcualte the pa mpjpe now
    left_hand_input_motion = result_data['left_hand_input_motion']
    right_hand_input_motion = result_data['right_hand_input_motion']
    mo2cap2_input_motion = result_data['mo2cap2_input_motion']
    left_hand_gt_motion = result_data['left_hand_gt_motion']
    right_hand_gt_motion = result_data['right_hand_gt_motion']
    mo2cap2_gt_motion = result_data['mo2cap2_gt_motion']

    # calcualte the pa mpjpe on the human body
    result = {}
    mo2cap2_mask = np.ones((mo2cap2_gt_motion.shape[0], mo2cap2_gt_motion.shape[1])).astype(bool)
    # smooth the input motion
    from scipy.ndimage import gaussian_filter1d
    mo2cap2_input_motion = gaussian_filter1d(mo2cap2_input_motion, sigma=3, axis=0)
    mo2cap2_input_mpjpe = keypoint_mpjpe(mo2cap2_input_motion, mo2cap2_gt_motion, mask=mo2cap2_mask,
                                         alignment="procrustes")

    result['mo2cap2_smooth_input_mpjpe'] = mo2cap2_input_mpjpe

    # calcualte the pa mpjpe on the left hand
    left_hand_mask = np.ones((left_hand_gt_motion.shape[0], left_hand_gt_motion.shape[1])).astype(bool)
    left_hand_input_motion = gaussian_filter1d(left_hand_input_motion, sigma=3, axis=0)
    left_hand_input_mpjpe = keypoint_mpjpe(left_hand_input_motion, left_hand_gt_motion,
                                           mask=left_hand_mask, alignment="procrustes")
    result['left_hand_smooth_input_mpjpe'] = left_hand_input_mpjpe

    # calcualte the pa mpjpe on the right hand
    right_hand_mask = np.ones((right_hand_gt_motion.shape[0], right_hand_gt_motion.shape[1])).astype(bool)
    right_hand_input_motion = gaussian_filter1d(right_hand_input_motion, sigma=3, axis=0)
    right_hand_input_mpjpe = keypoint_mpjpe(right_hand_input_motion, right_hand_gt_motion,
                                            mask=right_hand_mask, alignment="procrustes")
    result['right_hand_smooth_input_mpjpe'] = right_hand_input_mpjpe

    print(result)

    return result

def convert_with_ego_camera(pose_list, ego_camera_pose_list):
    result_list = []
    for body_pose, ego_camera_pose in zip(pose_list, ego_camera_pose_list):
        body_pose_homo = np.ones((body_pose.shape[0], 4))
        body_pose_homo[:, :3] = body_pose
        body_pose_homo = ego_camera_pose.dot(body_pose_homo.T).T
        body_pose_res = body_pose_homo[:, :3]
        result_list.append(body_pose_res)
    result_list = np.asarray(result_list)
    return result_list

def eval_motion_error(result_data):
    ego_camera_pose_list = result_data['ego_camera_pose_list']
    global_combined_input_motion = result_data['global_combined_input_motion']

    global_combined_output_motion = result_data['global_combined_output_motion']
    # convert the input and output motion
    # global_combined_output_motion = convert_with_ego_camera(global_combined_output_motion, ego_camera_pose_list)
    # global_combined_input_motion = convert_with_ego_camera(global_combined_input_motion, ego_camera_pose_list)
    left_hand_pred_motion = global_combined_output_motion[:, 15: 15 + 21]
    left_hand_input_motion = global_combined_input_motion[:, 15: 15 + 21]
    right_hand_pred_motion = global_combined_output_motion[:, 15 + 21: 15 + 21 + 21]
    right_hand_input_motion = global_combined_input_motion[:, 15 + 21: 15 + 21 + 21]
    mo2cap2_pred_motion = global_combined_output_motion[:, :15]
    mo2cap2_input_motion = global_combined_input_motion[:, :15]
    left_hand_gt_motion = result_data['left_hand_gt_motion']
    right_hand_gt_motion = result_data['right_hand_gt_motion']
    mo2cap2_gt_motion = result_data['mo2cap2_gt_motion']

    mo2cap2_gt_motion = convert_with_ego_camera(mo2cap2_gt_motion, ego_camera_pose_list)
    left_hand_gt_motion = convert_with_ego_camera(left_hand_gt_motion, ego_camera_pose_list)
    right_hand_gt_motion = convert_with_ego_camera(right_hand_gt_motion, ego_camera_pose_list)

    # for _ in range(0, 190, 10):
    #     mo2cap2_pred_mesh = draw_skeleton_with_chain(mo2cap2_pred_motion[_], mo2cap2_chain)
    #     mo2cap2_gt_mesh = draw_skeleton_with_chain(mo2cap2_gt_motion[_], mo2cap2_chain, bone_color=(1, 0, 0),
    #                                                joint_color=(1, 0, 0))
    #     mo2cap2_input_mesh = draw_skeleton_with_chain(mo2cap2_input_motion[_], mo2cap2_chain, bone_color=(0, 0, 1))
    #     open3d.visualization.draw_geometries([mo2cap2_pred_mesh, mo2cap2_gt_mesh, mo2cap2_input_mesh])

    # calcualte the pa mpjpe on the human body
    result = {}
    mo2cap2_pred_motion = mo2cap2_pred_motion - mo2cap2_pred_motion[:, 0:1] + mo2cap2_gt_motion[:, 0:1]
    mo2cap2_mask = np.ones((mo2cap2_gt_motion.shape[0], mo2cap2_gt_motion.shape[1])).astype(bool)
    mo2cap2_output_mpjpe = keypoint_mpjpe(mo2cap2_pred_motion, mo2cap2_gt_motion, mask=mo2cap2_mask, alignment='scale')
    mo2cap2_input_mpjpe = keypoint_mpjpe(mo2cap2_input_motion, mo2cap2_gt_motion, mask=mo2cap2_mask, alignment='scale')
    mo2cap2_output_pa_mpjpe = keypoint_mpjpe(mo2cap2_pred_motion, mo2cap2_gt_motion, mask=mo2cap2_mask, alignment="procrustes")
    mo2cap2_input_pa_mpjpe = keypoint_mpjpe(mo2cap2_input_motion, mo2cap2_gt_motion, mask=mo2cap2_mask, alignment="procrustes")
    result['mo2cap2_output_mpjpe'] = mo2cap2_output_mpjpe
    result['mo2cap2_input_mpjpe'] = mo2cap2_input_mpjpe
    result['mo2cap2_output_pa_mpjpe'] = mo2cap2_output_pa_mpjpe
    result['mo2cap2_input_pa_mpjpe'] = mo2cap2_input_pa_mpjpe

    # calcualte the pa mpjpe on the left hand
    left_hand_mask = np.ones((left_hand_gt_motion.shape[0], left_hand_gt_motion.shape[1])).astype(bool)
    left_hand_pred_motion = left_hand_pred_motion - left_hand_pred_motion[:, 0:1] + left_hand_gt_motion[:, 0:1]
    left_hand_input_motion = left_hand_input_motion - left_hand_input_motion[:, 0:1] + left_hand_gt_motion[:, 0:1]
    left_hand_output_mpjpe = keypoint_mpjpe(left_hand_pred_motion, left_hand_gt_motion,
                                            mask=left_hand_mask, alignment="scale")
    left_hand_output_pa_mpjpe = keypoint_mpjpe(left_hand_pred_motion, left_hand_gt_motion,
                                            mask=left_hand_mask, alignment="procrustes")
    left_hand_input_mpjpe = keypoint_mpjpe(left_hand_input_motion, left_hand_gt_motion,
                                           mask=left_hand_mask, alignment="scale")
    left_hand_input_pa_mpjpe = keypoint_mpjpe(left_hand_input_motion, left_hand_gt_motion,
                                           mask=left_hand_mask, alignment="procrustes")
    result['left_hand_output_mpjpe'] = left_hand_output_mpjpe
    result['left_hand_output_pa_mpjpe'] = left_hand_output_pa_mpjpe
    result['left_hand_input_mpjpe'] = left_hand_input_mpjpe
    result['left_hand_input_pa_mpjpe'] = left_hand_input_pa_mpjpe

    # calcualte the pa mpjpe on the right hand
    right_hand_mask = np.ones((right_hand_gt_motion.shape[0], right_hand_gt_motion.shape[1])).astype(bool)
    right_hand_pred_motion = right_hand_pred_motion - right_hand_pred_motion[:, 0:1] + right_hand_gt_motion[:, 0:1]
    right_hand_input_motion = right_hand_input_motion - right_hand_input_motion[:, 0:1] + right_hand_gt_motion[:, 0:1]
    right_hand_output_mpjpe = keypoint_mpjpe(right_hand_pred_motion, right_hand_gt_motion,
                                             mask=right_hand_mask, alignment="scale")
    right_hand_output_pa_mpjpe = keypoint_mpjpe(right_hand_pred_motion, right_hand_gt_motion,
                                             mask=right_hand_mask, alignment="procrustes")
    right_hand_input_mpjpe = keypoint_mpjpe(right_hand_input_motion, right_hand_gt_motion,
                                            mask=right_hand_mask, alignment="scale")
    right_hand_input_pa_mpjpe = keypoint_mpjpe(right_hand_input_motion, right_hand_gt_motion,
                                            mask=right_hand_mask, alignment="procrustes")
    result['right_hand_output_mpjpe'] = right_hand_output_mpjpe
    result['right_hand_output_pa_mpjpe'] = right_hand_output_pa_mpjpe
    result['right_hand_input_mpjpe'] = right_hand_input_mpjpe
    result['right_hand_input_pa_mpjpe'] = right_hand_input_pa_mpjpe

    print(result)

    return result

def visualize_input(data_id, eval_results, save_dir, render=False, save=False, save_viewpoint=False,
                      vis_skip_frame=1):
    view_point_save_path = os.path.join(save_dir, 'view_point.json')
    global_combined_input_motion = eval_results['global_combined_input_motion']
    mo2cap2_input_motion = global_combined_input_motion[:, :15]
    left_hand_input_motion = global_combined_input_motion[:, 15: 15 + 21]
    right_hand_input_motion = global_combined_input_motion[:, 15 + 21:]
    if render:
        render_image_save_dir = save_dir + '/' + f'render_imgs_input_{data_id}'
        # if os.path.exists(render_image_save_dir):
        #     return
        os.makedirs(render_image_save_dir, exist_ok=True)
    for i in range(0, len(mo2cap2_input_motion), vis_skip_frame):
        left_hand_pose = left_hand_input_motion[i]
        right_hand_pose = right_hand_input_motion[i]
        mo2cap2_pose = mo2cap2_input_motion[i]

        from mmpose.utils.visualization.draw import draw_skeleton_with_chain
        from mmpose.data.keypoints_mapping.mano import mano_skeleton
        from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
        from mmpose.utils.visualization.open3d_render_utils import render_open3d, save_view_point
        left_hand_mesh = draw_skeleton_with_chain(left_hand_pose, mano_skeleton, keypoint_radius=0.01,
                                                  line_radius=0.0025)
        right_hand_mesh = draw_skeleton_with_chain(right_hand_pose, mano_skeleton, keypoint_radius=0.01,
                                                   line_radius=0.0025)

        mo2cap2_mesh = draw_skeleton_with_chain(mo2cap2_pose, mo2cap2_chain)

        scene_mesh_path = r'X:\ScanNet\work\25-08-22\shot_bg\recon_bak2\ply\model.ply'
        scene_mesh = open3d.io.read_triangle_mesh(scene_mesh_path)
        scene_mesh = scene_mesh.scale(0.001, center=np.asarray([0, 0, 0]))

        overall_mesh = mo2cap2_mesh + left_hand_mesh + right_hand_mesh + scene_mesh

        coor = open3d.geometry.TriangleMesh.create_coordinate_frame()

        if save:
            mesh_save_path = os.path.join(save_dir, 'hand_%04d.ply' % i)
            open3d.io.write_triangle_mesh(mesh_save_path, overall_mesh)
        if save_viewpoint:
            save_view_point([overall_mesh, coor], view_point_save_path)
        if render:
            render_open3d([overall_mesh], view_point_save_path,
                          out_path=os.path.join(render_image_save_dir, 'hand_%04d.png' % i))

    if render:
        # use ffmpeg to render video
        # ffmpeg -r 30 -i hand_%04d.png -vcodec libx264 -pix_fmt yuv420p -crf 25 -y hand.mp4
        import subprocess
        import shlex
        subprocess.run(shlex.split(
            f'ffmpeg -r 25 -i {render_image_save_dir}/hand_%04d.png '
            f'-vcodec libx264 -pix_fmt yuv420p -y '
            f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
            f'{render_image_save_dir}/hand.mp4'))

def visualize_results(data_id, eval_results, save_dir, render=False, save=False, save_viewpoint=False, show=False,
                      vis_skip_frame=1):
    view_point_save_path = os.path.join(save_dir, 'view_point.json')
    global_combined_output_motion = eval_results['global_combined_output_motion']
    mo2cap2_pred_motion = global_combined_output_motion[:, :15]
    left_hand_pred_motion = global_combined_output_motion[:, 15: 15 + 21]
    right_hand_pred_motion = global_combined_output_motion[:, 15 + 21:]

    if save:
        save_obj_dir = save_dir + '/' + f'obj_{data_id}'
        os.makedirs(save_obj_dir, exist_ok=True)
    if render:
        render_image_save_dir = save_dir + '/' + f'render_imgs_{data_id}'
        # if os.path.exists(render_image_save_dir):
        #     return
        os.makedirs(render_image_save_dir, exist_ok=True)
    for i in range(0, len(mo2cap2_pred_motion), vis_skip_frame):
        left_hand_pose = left_hand_pred_motion[i]
        right_hand_pose = right_hand_pred_motion[i]
        mo2cap2_pose = mo2cap2_pred_motion[i]

        from mmpose.utils.visualization.draw import draw_skeleton_with_chain
        from mmpose.data.keypoints_mapping.mano import mano_skeleton
        from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
        from mmpose.utils.visualization.open3d_render_utils import render_open3d, save_view_point
        left_hand_mesh = draw_skeleton_with_chain(left_hand_pose, mano_skeleton, keypoint_radius=0.01,
                                                  line_radius=0.0025)
        right_hand_mesh = draw_skeleton_with_chain(right_hand_pose, mano_skeleton, keypoint_radius=0.01,
                                                   line_radius=0.0025)

        mo2cap2_mesh = draw_skeleton_with_chain(mo2cap2_pose, mo2cap2_chain)

        scene_mesh_path = r'X:\ScanNet\work\25-08-22\shot_bg\recon_bak2\ply\model.ply'
        scene_mesh = open3d.io.read_triangle_mesh(scene_mesh_path)
        scene_mesh = scene_mesh.scale(0.001, center=np.asarray([0, 0, 0]))

        overall_body_mesh = mo2cap2_mesh + left_hand_mesh + right_hand_mesh

        overall_mesh = overall_body_mesh + scene_mesh

        coor = open3d.geometry.TriangleMesh.create_coordinate_frame()
        if show:
            open3d.visualization.draw_geometries([overall_mesh, coor])

        if save:
            mesh_save_path = os.path.join(save_obj_dir, 'body_%04d.ply' % i)
            open3d.io.write_triangle_mesh(mesh_save_path, overall_body_mesh)

            mo2cap2_mesh_save_path = os.path.join(save_obj_dir, 'mo2cap2_body_%04d.ply' % i)
            open3d.io.write_triangle_mesh(mo2cap2_mesh_save_path, mo2cap2_mesh)
            left_hand_mesh_save_path = os.path.join(save_obj_dir, 'left_hand_%04d.ply' % i)
            open3d.io.write_triangle_mesh(left_hand_mesh_save_path, left_hand_mesh)
            right_hand_mesh_save_path = os.path.join(save_obj_dir, 'right_hand_%04d.ply' % i)
            open3d.io.write_triangle_mesh(right_hand_mesh_save_path, right_hand_mesh)
        if save_viewpoint:
            save_view_point([overall_mesh, coor], view_point_save_path)
        if render:
            render_open3d([overall_mesh], view_point_save_path,
                          out_path=os.path.join(render_image_save_dir, 'hand_%04d.png' % i))

    if render:
        # use ffmpeg to render video
        # ffmpeg -r 30 -i hand_%04d.png -vcodec libx264 -pix_fmt yuv420p -crf 25 -y hand.mp4
        import subprocess
        import shlex
        subprocess.run(shlex.split(
            f'ffmpeg -r 25 -i {render_image_save_dir}/hand_%04d.png '
            f'-vcodec libx264 -pix_fmt yuv420p -y '
            f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
            f'{render_image_save_dir}/hand.mp4'))


def run_whole_sequence_and_evaluate():
    # data_id_input = 47
    # main(data_id_input)
    # data_pkl_path = os.path.join(diffusion_result_save_dir, f"diffusion_results_{data_id_input}.pkl")
    # result_data = get_pred_gt_motion(data_pkl_path)
    # eval_motion_error(result_data)
    result_error_dict_list = []
    for data_id_input in range(0, 65):
        run_diffusion(data_id_input)
        data_pkl_path = os.path.join(diffusion_result_save_global_dir, f"diffusion_results_{data_id_input}.pkl")
        result_data = get_pred_gt_motion(data_pkl_path=data_pkl_path)
        # smooth_input_error_dict = eval_smooth_motion_error(result_data)
        data_pose_result_pkl_path = os.path.join(diffusion_result_save_global_dir,
                                                 f"diffusion_pose_results_{data_id_input}.pkl")
        with open(data_pose_result_pkl_path, 'wb') as f:
            pickle.dump(result_data, f)

        result_error_dict = eval_motion_error(result_data)
        data_eval_result_pkl_path = os.path.join(diffusion_result_save_global_dir,
                                                 f"diffusion_eval_results_{data_id_input}.pkl")
        with open(data_eval_result_pkl_path, 'wb') as f:
            pickle.dump(result_error_dict, f)
        # result_error_dict.update(smooth_input_error_dict)
        result_error_dict_list.append(result_error_dict)
        # visualize_results(data_pkl_path, save_viewpoint=True)
        # visualize_results(data_id_input, data_pkl_path, render=True)
    # average the error dict
    average_error_dict = {}
    for key in result_error_dict_list[0].keys():
        average_error_dict[key] = np.mean([result_error_dict[key] for result_error_dict in result_error_dict_list])
    print('average error is:')
    print(average_error_dict)

def visualize_reprojection_result(data_id, data_pkl_path, save_dir, render=False, save=False, save_viewpoint=False,):
    pass

def run_diffusion_and_visualize():
    data_id_input = 61
    diffusion_result = run_diffusion(data_id_input, save=False)
    data_pkl_path = os.path.join(diffusion_result_save_global_dir, f"diffusion_results_{data_id_input}.pkl")
    result_data = get_pred_gt_motion(data_pkl_path=None, diffusion_results_input=diffusion_result)
    data_pose_result_pkl_path = os.path.join(diffusion_result_save_global_dir,
                                             f"diffusion_pose_results_{data_id_input}.pkl")
    # with open(data_pose_result_pkl_path, 'wb') as f:
    #     pickle.dump(result_data, f)
    result_error_dict = eval_motion_error(result_data)
    data_eval_result_pkl_path = os.path.join(diffusion_result_save_global_dir,
                                             f"diffusion_eval_results_{data_id_input}.pkl")
    # with open(data_eval_result_pkl_path, 'wb') as f:
    #     pickle.dump(result_error_dict, f)

    visualize_results(data_id_input, result_data, save_dir=diffusion_result_save_global_dir, save_viewpoint=False,
                      render=False, save=False, show=True, vis_skip_frame=10)  # save the mesh
    # visualize_results(data_id_input, result_data, save_dir=diffusion_result_save_global_dir, save_viewpoint=False,
    #                   render=False, save=True)  # save the mesh
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
    # run_diffusion_and_visualize()
    run_whole_sequence_and_evaluate()