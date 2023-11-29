import json
import os
import pickle

import numpy as np
import open3d.visualization
from natsort import natsorted

from mmpose.data.keypoints_mapping.mano import mano_skeleton
from mmpose.utils.visualization.draw import draw_skeleton_with_chain
from mmpose.utils.visualization.open3d_render_utils import render_open3d, save_view_point

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def load_start_frame(syn_json):
    with open(syn_json, 'r') as f:
        data = json.load(f)
    return data['ego'], data['ext']


def load_diffusion_results(diffusion_result_dir):
    result_dict = {}
    eval_result_list = []
    for pkl_i in range(65):
        diffusion_eval_results = os.path.join(diffusion_result_dir, f'diffusion_eval_results_{pkl_i}.pkl')
        diffusion_results = os.path.join(diffusion_result_dir, f'diffusion_pose_results_{pkl_i}.pkl')
        with open(diffusion_eval_results, 'rb') as f:
            eval_results = pickle.load(f)

        with open(diffusion_results, 'rb') as f:
            diffusion_pose_results = pickle.load(f)

        seq_name = diffusion_pose_results['seq_name']
        seq_name = seq_name[4:]
        if seq_name not in result_dict:
            result_dict[seq_name] = {}
        ext_id_list = diffusion_pose_results['ext_id']
        eval_results['ext_id'] = ext_id_list
        eval_results['seq_name'] = seq_name
        eval_result_list.append(eval_results)
        for i in range(len(ext_id_list)):
            ext_id = ext_id_list[i]
            result_dict[seq_name][ext_id] = {k: v[i] for k, v in diffusion_pose_results.items()
                                             if k != 'ext_id' and k != 'seq_name'}
    return result_dict, eval_result_list


def convert_with_ego_camera(body_pose, ego_camera_pose):
    body_pose_homo = np.ones((body_pose.shape[0], 4))
    body_pose_homo[:, :3] = body_pose
    body_pose_homo = ego_camera_pose.dot(body_pose_homo.T).T
    body_pose_res = body_pose_homo[:, :3]
    return body_pose_res


def main():
    if os.name == 'nt':
        diffusion_result_dir = r'\\winfs-inf\CT\EgoMocap\work\EgocentricFullBody\vis_results\paper_vis_diffusion_results_hands'
    else:
        diffusion_result_dir = r'/CT/EgoMocap/work/EgocentricFullBody/vis_results/paper_vis_diffusion_results_hands'

    diffusion_pose_result, diffusion_mpjpe_result = load_diffusion_results(diffusion_result_dir)

    diffusion_mpjpe_result = [x for x in diffusion_mpjpe_result if x['left_hand_output_mpjpe'] < 0.018]
    diffusion_mpjpe_result_sorted = natsorted(diffusion_mpjpe_result,
                                              key=lambda x: x['left_hand_input_mpjpe'] - x['left_hand_output_mpjpe'] + x['right_hand_input_mpjpe'] - x['right_hand_output_mpjpe'],
                                              reverse=True)

    selected = diffusion_mpjpe_result_sorted[2]
    print(selected)

    seq_name = selected['seq_name']
    # ext_id = selected['ext_id'][50]
    camera_setup = False
    for ext_id in selected['ext_id']:
        # seq_name = 'jian2'
        print(f'start working on ext id: {ext_id}')
        if os.name == 'nt':
            base_path = fr'X:\ScanNet\work\egocentric_view\25082022\{seq_name}'
        else:
            base_path = fr'/HPS/ScanNet/work/egocentric_view/25082022/{seq_name}'

        with open(os.path.join(base_path, 'out', f'estimated_depth_new_{seq_name}_2.pkl'), 'rb') as f:
            scene_ego_data = pickle.load(f)

        with open(os.path.join(base_path, 'out', 'egopw_results_estimated_pose.pkl'), 'rb') as f:
            egopw_joint_dict = pickle.load(f)

        # with open(os.path.join(base_path, 'out', 'mo2cap2_results.pkl'), 'rb') as f:
        #     mo2cap2_pose_data = pickle.load(f)
        #
        # with open(os.path.join(base_path, 'out', 'xr_egopose_results.pkl'), 'rb') as f:
        #     xr_pose_data = pickle.load(f)

        with open(os.path.join(base_path, 'local_pose_gt.pkl'), 'rb') as f:
            local_pose_gt_data = pickle.load(f)

        diffusion_pose_result_seq_name = diffusion_pose_result[seq_name]
        # i = 1970
        # ext_id = 1700
        egocentric_start_frame, ext_start_frame = load_start_frame(os.path.join(base_path, 'syn.json'))
        i = ext_id - ext_start_frame
        ego_id = i + egocentric_start_frame
        # ext_id_calc = i + ext_start_frame
        # assert ext_id == ext_id_calc

        scene_ego_pose = scene_ego_data[i]

        pose_gt_item = local_pose_gt_data[i]

        ego_pose_gt = pose_gt_item['ego_pose_gt']
        # calib_board_pose = pose_gt_item['calib_board_pose']
        ext_id_pose_gt = pose_gt_item['ext_id']
        assert ext_id_pose_gt == ext_id

        img_path = os.path.join(base_path, 'imgs', 'img_%06d.jpg' % ego_id)

        diffusion_global_combined_output_joints = diffusion_pose_result_seq_name[ext_id][
            'global_combined_output_motion']
        diffusion_global_combined_input_joints = diffusion_pose_result_seq_name[ext_id]['global_combined_input_motion']
        ego_camera_pose = diffusion_pose_result_seq_name[ext_id]['ego_camera_pose_list']
        mo2cap2_gt_pose = diffusion_pose_result_seq_name[ext_id]['mo2cap2_gt_motion']

        egopw_joint = egopw_joint_dict['img_%06d.jpg' % ego_id]

        print(img_path)
        # mo2cap2_gt_pose = convert_with_ego_camera(mo2cap2_gt_pose, ego_camera_pose)
        # egopw_joint = convert_with_ego_camera(egopw_joint, ego_camera_pose)
        # scene_ego_pose = convert_with_ego_camera(scene_ego_pose, ego_camera_pose)
        # our_single_pose = diffusion_global_combined_input_joints[:15]
        # our_diffusion_pose = diffusion_global_combined_output_joints[:15]

        left_hand_hand4whole_pose = diffusion_pose_result_seq_name[ext_id]['left_hand_hand4whole_motion_global']
        left_hand_single_pose = diffusion_global_combined_input_joints[15:15 + 21]
        left_hand_diffusion_pose = diffusion_global_combined_output_joints[15:15 + 21]
        left_hand_gt_pose = diffusion_pose_result_seq_name[ext_id]['left_hand_gt_motion']
        left_hand_gt_pose = convert_with_ego_camera(left_hand_gt_pose, ego_camera_pose)

        # align them to the original point of the space
        left_hand_hand4whole_pose -= left_hand_hand4whole_pose[0:1]
        left_hand_single_pose -= left_hand_single_pose[0:1]
        left_hand_diffusion_pose -= left_hand_diffusion_pose[0:1]
        left_hand_gt_pose -= left_hand_gt_pose[0:1]

        left_hand_hand4whole_skeleton = draw_skeleton_with_chain(left_hand_hand4whole_pose, mano_skeleton,
                                                                 keypoint_radius=0.005,
                                                                 line_radius=0.0015
                                                                 )
        left_hand_single_skeleton = draw_skeleton_with_chain(left_hand_single_pose, mano_skeleton,
                                                             keypoint_radius=0.005,
                                                             line_radius=0.0015
                                                             )
        left_hand_diffusion_skeleton = draw_skeleton_with_chain(left_hand_diffusion_pose, mano_skeleton,
                                                                keypoint_radius=0.005,
                                                                line_radius=0.0015
                                                                )
        left_hand_gt_skeleton = draw_skeleton_with_chain(left_hand_gt_pose, mano_skeleton, joint_color=(1, 0, 0),
                                                         bone_color=(1, 0, 0),
                                                         keypoint_radius=0.005,
                                                         line_radius=0.0015
                                                         )

        right_hand_hand4whole_pose = diffusion_pose_result_seq_name[ext_id]['right_hand_hand4whole_motion_global']
        right_hand_single_pose = diffusion_global_combined_input_joints[15 + 21:]
        right_hand_diffusion_pose = diffusion_global_combined_output_joints[15 + 21:]
        right_hand_gt_pose = diffusion_pose_result_seq_name[ext_id]['right_hand_gt_motion']
        right_hand_gt_pose = convert_with_ego_camera(right_hand_gt_pose, ego_camera_pose)

        right_hand_diffusion_pose -= right_hand_diffusion_pose[0:1]
        right_hand_single_pose -= right_hand_single_pose[0:1]
        right_hand_hand4whole_pose -= right_hand_hand4whole_pose[0:1]
        right_hand_gt_pose -= right_hand_gt_pose[0:1]

        right_hand_hand4whole_skeleton = draw_skeleton_with_chain(right_hand_hand4whole_pose, mano_skeleton,
                                                                  keypoint_radius=0.005,
                                                                  line_radius=0.0015
                                                                  )
        right_hand_single_skeleton = draw_skeleton_with_chain(right_hand_single_pose, mano_skeleton,
                                                              keypoint_radius=0.005,
                                                              line_radius=0.0015
                                                              )
        right_hand_diffusion_skeleton = draw_skeleton_with_chain(right_hand_diffusion_pose, mano_skeleton,
                                                                 keypoint_radius=0.005,
                                                                 line_radius=0.0015
                                                                 )
        right_hand_gt_skeleton = draw_skeleton_with_chain(right_hand_gt_pose, mano_skeleton, joint_color=(1, 0, 0),
                                                          bone_color=(1, 0, 0),
                                                          keypoint_radius=0.005,
                                                          line_radius=0.0015
                                                          )

        # rendering
        left_hand_view_point_save_path = os.path.join(diffusion_result_dir, 'left_hand_view_point.json')
        right_hand_view_point_save_path = os.path.join(diffusion_result_dir, 'right_hand_view_point.json')
        ext_id_start = selected['ext_id'][0]
        ext_id_end = selected['ext_id'][-1]

        if not camera_setup:
            coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            save_view_point([left_hand_hand4whole_skeleton, left_hand_gt_skeleton, coord],
                            left_hand_view_point_save_path)
            save_view_point([right_hand_hand4whole_skeleton, right_hand_gt_skeleton, coord],
                            right_hand_view_point_save_path)
            camera_setup = True

        left_render_image_hand4whole_save_dir = diffusion_result_dir + '/' + f'left_hand_render_imgs_hand4whole_{seq_name}_{ext_id_start}'
        os.makedirs(left_render_image_hand4whole_save_dir, exist_ok=True)

        left_render_image_single_save_dir = diffusion_result_dir + '/' + f'left_hand_render_imgs_single_{seq_name}_{ext_id_start}'
        os.makedirs(left_render_image_single_save_dir, exist_ok=True)

        left_render_image_refined_save_dir = diffusion_result_dir + '/' + f'left_hand_render_imgs_refined_{seq_name}_{ext_id_start}'
        os.makedirs(left_render_image_refined_save_dir, exist_ok=True)

        # for right hand
        right_render_image_hand4whole_save_dir = diffusion_result_dir + '/' + f'right_hand_render_imgs_hand4whole_{seq_name}_{ext_id_start}'
        os.makedirs(right_render_image_hand4whole_save_dir, exist_ok=True)

        right_render_image_single_save_dir = diffusion_result_dir + '/' + f'right_hand_render_imgs_single_{seq_name}_{ext_id_start}'
        os.makedirs(right_render_image_single_save_dir, exist_ok=True)

        right_render_image_refined_save_dir = diffusion_result_dir + '/' + f'right_hand_render_imgs_refined_{seq_name}_{ext_id_start}'
        os.makedirs(right_render_image_refined_save_dir, exist_ok=True)



        render_open3d([left_hand_hand4whole_skeleton, left_hand_gt_skeleton], left_hand_view_point_save_path,
                      out_path=os.path.join(left_render_image_hand4whole_save_dir, 'img_%06d.png' % ego_id))
        open3d.io.write_triangle_mesh(os.path.join(left_render_image_hand4whole_save_dir, 'mesh_%06d.ply' % ego_id),
                                      left_hand_hand4whole_skeleton + left_hand_gt_skeleton)

        render_open3d([left_hand_single_skeleton, left_hand_gt_skeleton], left_hand_view_point_save_path,
                      out_path=os.path.join(left_render_image_single_save_dir, 'img_%06d.png' % ego_id))

        open3d.io.write_triangle_mesh(os.path.join(left_render_image_single_save_dir, 'mesh_%06d.ply' % ego_id),
                                      left_hand_single_skeleton + left_hand_gt_skeleton)

        render_open3d([left_hand_diffusion_skeleton, left_hand_gt_skeleton], left_hand_view_point_save_path,
                      out_path=os.path.join(left_render_image_refined_save_dir, 'img_%06d.png' % ego_id))

        open3d.io.write_triangle_mesh(os.path.join(left_render_image_refined_save_dir, 'mesh_%06d.ply' % ego_id),
                                      left_hand_diffusion_skeleton + left_hand_gt_skeleton)

        # right hand
        render_open3d([right_hand_hand4whole_skeleton, right_hand_gt_skeleton], right_hand_view_point_save_path,
                      out_path=os.path.join(right_render_image_hand4whole_save_dir, 'img_%06d.png' % ego_id))
        open3d.io.write_triangle_mesh(os.path.join(right_render_image_hand4whole_save_dir, 'mesh_%06d.ply' % ego_id),
                                      right_hand_hand4whole_skeleton + right_hand_gt_skeleton)

        render_open3d([right_hand_single_skeleton, right_hand_gt_skeleton], right_hand_view_point_save_path,
                      out_path=os.path.join(right_render_image_single_save_dir, 'img_%06d.png' % ego_id))

        open3d.io.write_triangle_mesh(os.path.join(right_render_image_single_save_dir, 'mesh_%06d.ply' % ego_id),
                                      right_hand_single_skeleton + right_hand_gt_skeleton)

        render_open3d([right_hand_diffusion_skeleton, right_hand_gt_skeleton], right_hand_view_point_save_path,
                      out_path=os.path.join(right_render_image_refined_save_dir, 'img_%06d.png' % ego_id))

        open3d.io.write_triangle_mesh(os.path.join(right_render_image_refined_save_dir, 'mesh_%06d.ply' % ego_id),
                                      right_hand_diffusion_skeleton + right_hand_gt_skeleton)


if __name__ == '__main__':
    main()
