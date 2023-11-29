#  Copyright Jian Wang @ MPI-INF (c) 2023.
import os
import pickle
from copy import deepcopy

import numpy as np
import open3d
import torch
from natsort import natsorted
from tqdm import tqdm

from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
from mmpose.utils.visualization.draw import draw_skeleton_with_chain


def visualize_joints(egopw_joints, sceneego_joints, our_single_joints, our_diff_joints):
    egopw_mesh = draw_skeleton_with_chain(egopw_joints, mo2cap2_chain, joint_color=(1, 0, 0),
                                          bone_color=(1, 0, 0))
    sceneego_mesh = draw_skeleton_with_chain(sceneego_joints, mo2cap2_chain, joint_color=(0, 1, 0),
                                             bone_color=(0, 1, 0))
    our_single_mesh = draw_skeleton_with_chain(our_single_joints, mo2cap2_chain, joint_color=(0, 0, 1),
                                               bone_color=(0, 0, 1))
    our_diff_mesh = draw_skeleton_with_chain(our_diff_joints, mo2cap2_chain)

    open3d.visualization.draw_geometries([egopw_mesh, sceneego_mesh, our_single_mesh, our_diff_mesh])


def render_joints(save_dir, image_file_name,
                  human_name, seq_name,
                  save=True, save_viewpoint=True,
                  hand4whole_left_hand=None, hand4whole_right_hand=None,
                  our_single_left_hand=None, our_single_right_hand=None,
                  our_diffusion_left_hand=None, our_diffusion_right_hand=None):
    # render_image_save_dir = save_dir + '/' + f'render_imgs_{human_name}_{seq_name}_{image_file_name}'
    render_image_save_dir = save_dir + '/' + f'render_imgs_{human_name}_{seq_name}'
    os.makedirs(render_image_save_dir, exist_ok=True)

    from mmpose.utils.visualization.draw import draw_skeleton_with_chain
    from mmpose.data.keypoints_mapping.mano import mano_skeleton
    from mmpose.utils.visualization.open3d_render_utils import render_open3d, save_view_point

    hand4whole_left_hand = hand4whole_left_hand - hand4whole_left_hand[0]
    hand4whole_right_hand = hand4whole_right_hand - hand4whole_right_hand[0]
    hand4whole_lh_mesh = draw_skeleton_with_chain(hand4whole_left_hand, mano_skeleton, keypoint_radius=0.005,
                                                  line_radius=0.0015)
    hand4whole_rh_mesh = draw_skeleton_with_chain(hand4whole_right_hand, mano_skeleton, keypoint_radius=0.005,
                                                  line_radius=0.0015)

    if our_single_left_hand is not None:
        our_single_left_hand = our_single_left_hand - our_single_left_hand[0]
        our_single_lh_mesh = draw_skeleton_with_chain(our_single_left_hand, mano_skeleton, keypoint_radius=0.005,
                                                      line_radius=0.0015)
        our_single_right_hand = our_single_right_hand - our_single_right_hand[0]
        our_single_rh_mesh = draw_skeleton_with_chain(our_single_right_hand, mano_skeleton, keypoint_radius=0.005,
                                                      line_radius=0.0015)
    if our_diffusion_left_hand is not None:
        our_diffusion_left_hand[0] *= 0
        our_diffusion_left_hand = our_diffusion_left_hand
        our_diffusion_lh_mesh = draw_skeleton_with_chain(our_diffusion_left_hand, mano_skeleton, keypoint_radius=0.005,
                                                         line_radius=0.0015)
        our_diffusion_right_hand[0] *= 0
        our_diffusion_right_hand = our_diffusion_right_hand
        our_diffusion_rh_mesh = draw_skeleton_with_chain(our_diffusion_right_hand, mano_skeleton, keypoint_radius=0.005,
                                                         line_radius=0.0015)
    left_hand_view_point_save_path = os.path.join(render_image_save_dir, 'left_hand_view_point.json')
    right_hand_view_point_save_path = os.path.join(render_image_save_dir, 'right_hand_view_point.json')
    if save_viewpoint:
        save_view_point([our_diffusion_lh_mesh, our_single_lh_mesh], left_hand_view_point_save_path)
        save_view_point([our_diffusion_rh_mesh, our_single_rh_mesh], right_hand_view_point_save_path)

    # render_open3d([egopw_mesh], view_point_save_path,
    #               out_path=os.path.join(render_image_save_dir, 'egopw.png'))
    hand4whole_save_dir = os.path.join(render_image_save_dir, 'hand4whole')
    os.makedirs(hand4whole_save_dir, exist_ok=True)
    render_open3d([hand4whole_lh_mesh], left_hand_view_point_save_path,
                  out_path=os.path.join(hand4whole_save_dir, 'left' + image_file_name))
    render_open3d([hand4whole_rh_mesh], right_hand_view_point_save_path,
                  out_path=os.path.join(hand4whole_save_dir, 'right' + image_file_name))

    our_single_save_dir = os.path.join(render_image_save_dir, 'single')
    os.makedirs(our_single_save_dir, exist_ok=True)
    render_open3d([our_single_lh_mesh], left_hand_view_point_save_path,
                  out_path=os.path.join(our_single_save_dir, 'left' + image_file_name))
    render_open3d([our_single_rh_mesh], right_hand_view_point_save_path,
                  out_path=os.path.join(our_single_save_dir, 'right' + image_file_name))

    our_refined_save_dir = os.path.join(render_image_save_dir, 'refined')
    os.makedirs(our_refined_save_dir, exist_ok=True)
    render_open3d([our_diffusion_lh_mesh], left_hand_view_point_save_path,
                  out_path=os.path.join(our_refined_save_dir, 'left' + image_file_name))
    render_open3d([our_diffusion_rh_mesh], right_hand_view_point_save_path,
                  out_path=os.path.join(our_refined_save_dir, 'right' + image_file_name))

    if save:
        # open3d.io.write_triangle_mesh(os.path.join(render_image_save_dir, 'egopw.ply'), egopw_mesh)
        open3d.io.write_triangle_mesh(os.path.join(hand4whole_save_dir, 'left_{}.ply'.format(image_file_name)),
                                      hand4whole_lh_mesh)
        open3d.io.write_triangle_mesh(os.path.join(hand4whole_save_dir, 'right_{}.ply'.format(image_file_name)),
                                      hand4whole_rh_mesh)
        open3d.io.write_triangle_mesh(os.path.join(our_single_save_dir, 'left_{}.ply'.format(image_file_name)),
                                      our_single_lh_mesh)
        open3d.io.write_triangle_mesh(os.path.join(our_single_save_dir, 'right_{}.ply'.format(image_file_name)),
                                      our_single_rh_mesh)
        open3d.io.write_triangle_mesh(os.path.join(our_refined_save_dir, 'left_{}.ply'.format(image_file_name)),
                                      our_diffusion_lh_mesh)
        open3d.io.write_triangle_mesh(os.path.join(our_refined_save_dir, 'right_{}.ply'.format(image_file_name)),
                                      our_diffusion_rh_mesh)


def read_out_diffusion_result(pkl_dir, mean_std_path):
    pkl_name_list = natsorted(os.listdir(pkl_dir))
    pkl_name_list = [pkl_name for pkl_name in pkl_name_list if pkl_name.endswith('.pkl')]
    data_dict = {}
    for pkl_name in tqdm(pkl_name_list):
        pkl_path = os.path.join(pkl_dir, pkl_name)
        with open(pkl_path, 'rb') as f:
            diffusion_results = pickle.load(f)
        normalize = True

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

        left_hand_motion_list = left_hand_motion_sequence_list.reshape(196, 21, 3)
        right_hand_motion_list = right_hand_motion_sequence_list.reshape(196, 21, 3)
        mo2cap2_motion_list = mo2cap2_motion_sequence_list.reshape(196, 15, 3)

        # get image meta information
        img_metas = diffusion_results['img_metas'].data
        img_path_list = img_metas['image_path']
        seq_name_list = img_metas['seq_name']
        human_name_list = img_metas['human_name']
        name_list = img_metas['name']
        for i in range(len(img_path_list)):
            image_file_path = img_path_list[i]
            human_name = image_file_path.split('/')[-5] + '_' + image_file_path.split('/')[-4]
            seq_name = image_file_path.split('/')[-3]
            image_name = image_file_path.split('/')[-1]
            if human_name not in data_dict.keys():
                data_dict[human_name] = {}
            if seq_name not in data_dict[human_name].keys():
                data_dict[human_name][seq_name] = []

            pred_joints_3d = mo2cap2_motion_list[i]
            pred_left_hand_joint_3d = left_hand_motion_list[i]
            pred_right_hand_joint_3d = right_hand_motion_list[i]
            data_dict[human_name][seq_name].append({
                'image_name': image_name,
                'image_path': image_file_path,
                'pred_joints_3d': pred_joints_3d,
                'pred_left_hand_joint_3d': pred_left_hand_joint_3d,
                'pred_right_hand_joint_3d': pred_right_hand_joint_3d
            })
    return data_dict


def read_our_singleframe_result(pkl_path):
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

    data_dict = {

    }
    # split by the id name and seq name
    for i in range(len(image_file_path_list)):
        image_file_path = image_file_path_list[i]
        human_name = image_file_path.split('/')[-5] + '_' + image_file_path.split('/')[-4]
        seq_name = image_file_path.split('/')[-3]
        image_name = image_file_path.split('/')[-1]
        if human_name not in data_dict.keys():
            data_dict[human_name] = {}
        if seq_name not in data_dict[human_name].keys():
            data_dict[human_name][seq_name] = []

        pred_joints_3d = pred_joints_3d_list[i]
        pred_left_hand_joint_3d = pred_left_hand_joint_3d_list[i]
        pred_right_hand_joint_3d = pred_right_hand_joint_3d_list[i]
        data_dict[human_name][seq_name].append({
            'image_name': image_name,
            'image_path': image_file_path,
            'pred_joints_3d': pred_joints_3d,
            'pred_left_hand_joint_3d': pred_left_hand_joint_3d,
            'pred_right_hand_joint_3d': pred_right_hand_joint_3d
        })
    return data_dict


def main():
    if os.name == 'nt':
        hand4whole_pkl_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_egopw_orig_hand_egopw_body\outputs.pkl'
        pkl_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_egopw\outputs.pkl'
        diffusion_result_dir = r'Z:\EgoMocap\work\EgocentricFullBody\vis_results\egopw_diffusion_results_new'
        mean_std_path = r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\diffusion_fullbody\ego_mean_std.pkl'
        data_dir = r'X:\Mo2Cap2Plus1\static00\ExternalEgo'
        save_dir = r'Z:\EgoMocap\work\EgocentricFullBody\vis_results\egopw_diffusion_results_hands'
    else:
        hand4whole_pkl_path = r'/CT/EgoMocap/work/EgocentricFullBody/work_dirs/egofullbody_test_egopw_orig_hand_egopw_body/outputs.pkl'
        pkl_path = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/egofullbody_test_egopw/outputs.pkl'
        data_dir = '/CT/Mo2Cap2Plus1/static00/ExternalEgo'
        diffusion_result_dir = r'/CT/EgoMocap/work/EgocentricFullBody/vis_results/egopw_diffusion_results_new'
        mean_std_path = r'/CT/EgoMocap/work/EgocentricFullBody/dataset_files/diffusion_fullbody/ego_mean_std.pkl'
        save_dir = '/CT/EgoMocap/work/EgocentricFullBody/vis_results/egopw_diffusion_results_hands'

    our_single_data_dict = read_our_singleframe_result(pkl_path)
    our_diff_data_dict = read_out_diffusion_result(diffusion_result_dir, mean_std_path)
    hand4whole_data_dict = read_our_singleframe_result(hand4whole_pkl_path)
    dir_name = 'External_camera_new'
    human_name = 'lingjie'
    data_dict_human_name = dir_name + '_' + human_name
    seq_name = 'out1'
    save_viewpoint = True
    # image_id = 100
    for i in range(1000, 2000, 10):
        # if i != 1707:
        #     continue
        hand4whole_result = hand4whole_data_dict[data_dict_human_name][seq_name]
        our_single_result = our_single_data_dict[data_dict_human_name][seq_name]
        our_diff_result = our_diff_data_dict[data_dict_human_name][seq_name]
        our_single_result = natsorted(our_single_result, key=lambda x: x['image_name'])
        our_single_result_list = [res['pred_joints_3d'] for res in our_single_result]

        hand4whole_result = natsorted(hand4whole_result, key=lambda x: x['image_name'])
        from scipy.ndimage import gaussian_filter1d
        smoothed_result_list = gaussian_filter1d(our_single_result_list, sigma=1, axis=0)
        our_diff_result = natsorted(our_diff_result, key=lambda x: x['image_name'])
        image_name = our_single_result[i]['image_name']
        print('id: {}_{}_{}_{}'.format(i, data_dict_human_name, seq_name, image_name))
        our_single_pose = our_single_result[i]['pred_joints_3d']
        our_diff_pose = our_diff_result[i]['pred_joints_3d']
        smoothed_pose = smoothed_result_list[i]

        image_id = int(image_name.split('.')[0].split('_')[-1])

        # visualize_joints(egopw_pose, sceneego_pose, our_single_pose, our_diff_pose)
        render_joints(save_dir=save_dir,
                      human_name=dir_name + human_name, seq_name=seq_name, image_file_name=image_name, save=True,
                      save_viewpoint=save_viewpoint,
                      hand4whole_left_hand=hand4whole_result[i]['pred_left_hand_joint_3d'],
                      hand4whole_right_hand=hand4whole_result[i]['pred_right_hand_joint_3d'],
                      our_single_left_hand=our_single_result[i]['pred_left_hand_joint_3d'],
                      our_single_right_hand=our_single_result[i]['pred_right_hand_joint_3d'],
                      our_diffusion_left_hand=our_diff_result[i]['pred_left_hand_joint_3d'],
                      our_diffusion_right_hand=our_diff_result[i]['pred_right_hand_joint_3d'])
        if save_viewpoint is True:
            save_viewpoint = False


if __name__ == '__main__':
    main()
