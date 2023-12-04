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
from mmpose.datasets.pipelines import (Collect,
                                       PreProcessHandMotion, AlignGlobalSMPLXJoints, SplitGlobalSMPLXJoints,
                                       PreProcessMo2Cap2BodyMotion)
if os.name == "nt":
    network_pred_seq_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_globalegomocap\results.pkl'
    mean_std_path = r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\diffusion_fullbody\ego_mean_std.pkl'
    diffusion_model_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\diffusion_full_body_train_uncond_ego_coord\epoch_12.pth'
    diffusion_result_save_dir = r"Z:/EgoMocap/work/EgocentricFullBody/vis_results/egofullbody_diffusion_results_ego_coord"
else:
    network_pred_seq_path = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/egofullbody_test_globalegomocap/results.pkl'
    mean_std_path = '/CT/EgoMocap/work/EgocentricFullBody/dataset_files/diffusion_fullbody/ego_mean_std.pkl'
    diffusion_model_path = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/diffusion_full_body_train_uncond_ego_coord/epoch_12.pth'
    diffusion_result_save_dir = "/CT/EgoMocap/work/EgocentricFullBody/vis_results/egofullbody_diffusion_results_ego_coord"


def get_fullbody_motion_test_dataset():
    seq_len = 196
    normalize = True
    pipeline = [
        SplitGlobalSMPLXJoints(smplx_joint_name='ego_smplx_joints'),
        PreProcessHandMotion(normalize=normalize,
                             mean_std_path=mean_std_path),
        PreProcessMo2Cap2BodyMotion(normalize=normalize,
                                    mean_std_path=mean_std_path),
        Collect(keys=['mo2cap2_body_features', 'left_hand_features', 'right_hand_features',
                      'processed_left_hand_keypoints_3d', 'processed_right_hand_keypoints_3d'],
                meta_keys=['gt_joints_3d'])
    ]

    dataset_cfg = dict(
        type='FullBodyEgoMotionTestDataset',
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

    # save the results
    if not os.path.exists(diffusion_result_save_dir):
        os.makedirs(diffusion_result_save_dir)
    with open(os.path.join(diffusion_result_save_dir, f"diffusion_results_{data_id}.pkl"), 'wb') as f:
        pickle.dump(diffusion_results, f)

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
    mo2cap2_gt_motion = img_metas.data['gt_joints_3d']


    # visualize for debug
    # from mmpose.utils.visualization.draw import draw_skeleton_with_chain
    # from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
    # from mmpose.data.keypoints_mapping.mano import mano_skeleton
    # mo2cap2_gt_mesh = draw_skeleton_with_chain(mo2cap2_gt_motion[0], mo2cap2_chain)
    # left_hand_gt_mesh = draw_skeleton_with_chain(left_hand_gt_motion[0], mano_skeleton, keypoint_radius=0.01,
    #                                                   line_radius=0.0025)
    # left_hand_pred_mesh = draw_skeleton_with_chain(left_hand_pred_motion[0], mano_skeleton, keypoint_radius=0.01,
    #                                                   line_radius=0.0025)
    # coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
    # open3d.visualization.draw_geometries([left_hand_gt_mesh])
    # open3d.visualization.draw_geometries([left_hand_pred_mesh])

    result = {'left_hand_pred_motion': left_hand_pred_motion,
              'left_hand_input_motion': left_hand_input_motion,
              'right_hand_pred_motion': right_hand_pred_motion,
              'right_hand_input_motion': right_hand_input_motion,
              'mo2cap2_pred_motion': mo2cap2_pred_motion,
              'mo2cap2_input_motion': mo2cap2_input_motion,
              'mo2cap2_gt_motion': mo2cap2_gt_motion,
              }
    return result



def eval_motion_error(result_data):
    # only calcualte the pa mpjpe now
    mo2cap2_pred_motion = result_data['mo2cap2_pred_motion']
    mo2cap2_input_motion = result_data['mo2cap2_input_motion']
    mo2cap2_gt_motion = result_data['mo2cap2_gt_motion']

    # calcualte the pa mpjpe on the human body
    result = {}
    mo2cap2_mask = np.ones((mo2cap2_gt_motion.shape[0], mo2cap2_gt_motion.shape[1])).astype(bool)
    mo2cap2_output_mpjpe = keypoint_mpjpe(mo2cap2_pred_motion, mo2cap2_gt_motion, mask=mo2cap2_mask, alignment="procrustes")
    mo2cap2_input_mpjpe = keypoint_mpjpe(mo2cap2_input_motion, mo2cap2_gt_motion, mask=mo2cap2_mask, alignment="procrustes")
    result['mo2cap2_output_mpjpe'] = mo2cap2_output_mpjpe
    result['mo2cap2_input_mpjpe'] = mo2cap2_input_mpjpe

    print(result)
    return result


def visualize_results(data_id, data_pkl_path, render=False, save=False, save_viewpoint=False,
                      vis_skip_frame=1, only_hand=False):
    with open(data_pkl_path, 'rb') as f:
        diffusion_results = pickle.load(f)
    normalize = True
    save_dir = os.path.dirname(data_pkl_path)
    view_point_save_path = os.path.join(save_dir, 'view_point.json')

    full_body_motion_sequence_list = diffusion_results['sample'].cpu().numpy()
    mo2cap2_motion_sequence_list = full_body_motion_sequence_list[:, :, :15 * 3]
    left_hand_motion_sequence_list = full_body_motion_sequence_list[:, :, 15 * 3: 15 * 3 + 21 * 3]
    right_hand_motion_sequence_list = full_body_motion_sequence_list[:, :, 15 * 3 + 21 * 3:]

    with open(mean_std_path, 'rb') as f:
        global_aligned_mean_std = pickle.load(f)

    if normalize:
        left_hand_mean = global_aligned_mean_std['left_hand_mean']
        left_hand_std = global_aligned_mean_std['left_hand_std']
        left_hand_motion_sequence_list = left_hand_motion_sequence_list * left_hand_std + left_hand_mean
        right_hand_mean = global_aligned_mean_std['right_hand_mean']
        right_hand_std = global_aligned_mean_std['right_hand_std']
        right_hand_motion_sequence_list = right_hand_motion_sequence_list * right_hand_std + right_hand_mean
        mo2cap2_body_mean = global_aligned_mean_std['mo2cap2_body_mean']
        mo2cap2_body_std = global_aligned_mean_std['mo2cap2_body_std']
        mo2cap2_motion_sequence_list = mo2cap2_motion_sequence_list * mo2cap2_body_std + mo2cap2_body_mean

    left_hand_motion_list = left_hand_motion_sequence_list.reshape(-1, 196, 21, 3)
    right_hand_motion_list = right_hand_motion_sequence_list.reshape(-1, 196, 21, 3)
    mo2cap2_motion_list = mo2cap2_motion_sequence_list.reshape(-1, 196, 15, 3)
    for motion_i, (left_hand_motion, right_hand_motion, mo2cap2_motion) in enumerate(zip(left_hand_motion_list,
                                                                                         right_hand_motion_list,
                                                                                         mo2cap2_motion_list)):

        left_hand_motion[:, 0] *= 0
        right_hand_motion[:, 0] *= 0

        assert len(left_hand_motion) == len(mo2cap2_motion) == len(right_hand_motion)

        if render:
            render_image_save_dir = save_dir + '/' + f'render_imgs_{data_id}'
            if os.path.exists(render_image_save_dir):
                continue
            os.makedirs(render_image_save_dir, exist_ok=True)
        for i in range(0, len(left_hand_motion), vis_skip_frame):
            left_hand_pose = left_hand_motion[i]
            right_hand_pose = right_hand_motion[i]
            mo2cap2_pose = mo2cap2_motion[i]
            if not only_hand:
                # combine left hand and mo2cap2 pose
                left_hand_root = mo2cap2_pose[6: 7]
                left_hand_pose += left_hand_root

                right_hand_root = mo2cap2_pose[3: 4]
                right_hand_pose += right_hand_root

            from mmpose.utils.visualization.draw import draw_skeleton_with_chain
            from mmpose.data.keypoints_mapping.mano import mano_skeleton
            from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
            from mmpose.utils.visualization.open3d_render_utils import render_open3d, save_view_point
            left_hand_mesh = draw_skeleton_with_chain(left_hand_pose, mano_skeleton, keypoint_radius=0.01,
                                                      line_radius=0.0025)
            right_hand_mesh = draw_skeleton_with_chain(right_hand_pose, mano_skeleton, keypoint_radius=0.01,
                                                       line_radius=0.0025)

            mo2cap2_mesh = draw_skeleton_with_chain(mo2cap2_pose, mo2cap2_chain)

            if only_hand:
                overall_mesh = left_hand_mesh
            else:
                overall_mesh = mo2cap2_mesh + left_hand_mesh + right_hand_mesh

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


def run_whole_sequence_and_evaluate():
    # data_id_input = 47
    # main(data_id_input)
    # data_pkl_path = os.path.join(diffusion_result_save_dir, f"diffusion_results_{data_id_input}.pkl")
    # result_data = get_pred_gt_motion(data_pkl_path)
    # eval_motion_error(result_data)
    result_error_dict_list = []
    for data_id_input in range(0, 59):
        run_diffusion(data_id_input)
        data_pkl_path = os.path.join(diffusion_result_save_dir, f"diffusion_results_{data_id_input}.pkl")
        if not os.path.exists(data_pkl_path):
            continue
        result_data = get_pred_gt_motion(data_pkl_path)
        result_error_dict = eval_motion_error(result_data)
        result_error_dict_list.append(result_error_dict)
        # visualize_results(data_pkl_path, save_viewpoint=True)
        # visualize_results(data_id_input, data_pkl_path, render=True)
    # average the error dict
    average_error_dict = {}
    for key in result_error_dict_list[0].keys():
        average_error_dict[key] = np.mean([result_error_dict[key] for result_error_dict in result_error_dict_list])
    print('average error is:')
    print(average_error_dict)

def run_diffusion_and_visualize():
    data_id_input = 47
    run_diffusion(data_id_input)
    data_pkl_path = os.path.join(diffusion_result_save_dir, f"diffusion_results_{data_id_input}.pkl")
    visualize_results(data_id_input, data_pkl_path, save_viewpoint=True)

if __name__ == '__main__':
    # run_diffusion_and_visualize()
    run_whole_sequence_and_evaluate()