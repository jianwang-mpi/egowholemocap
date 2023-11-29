#  Copyright Jian Wang @ MPI-INF (c) 2023.
import os.path
import pickle

import numpy as np
import open3d

from mmpose.data.keypoints_mapping.mano import mano_skeleton
from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
from mmpose.utils.visualization.draw import draw_skeleton_with_chain
from mmpose.utils.visualization.open3d_render_utils import render_open3d, save_view_point

result_path = r'Z:/EgoMocap/work/EgocentricFullBody/work_dirs/diffusion_full_body_train_uncond_ego_coord/outputs_eval_12.pkl'
normalize = True
vis_skip_frame = 1
save = False
vis = False

save_dir = r'Z:/EgoMocap/work/EgocentricFullBody/vis_results/diffusion_full_body_train_uncond_ego_coord/'
os.makedirs(save_dir, exist_ok=True)
view_point_save_path = os.path.join(save_dir, 'view_point.json')
render = True
save_viewpoint = False
only_hand = False
motion_id = [10, 20, 30, 80, 90, 100, 110]

with open(result_path, 'rb') as f:
    results = pickle.load(f)

result_motion_sequence_list = []
diffusion_process_list = []
for result_block in results:
    result_motion_sequence_list.extend(result_block['sample'].cpu().numpy())

    # diffusion_process_batch = result_block['diffusion_list']
    # diffusion_batch_out = np.asarray(diffusion_process_batch)
    # diffusion_batch_out = np.transpose(diffusion_batch_out, (1, 0, 2, 3))
    # diffusion_process_list.extend(diffusion_batch_out)

full_body_motion_sequence_list = np.asarray(result_motion_sequence_list)

# split mo2cap2 body, left hand and right hand
mo2cap2_motion_sequence_list = full_body_motion_sequence_list[:, :, :15 * 3]
left_hand_motion_sequence_list = full_body_motion_sequence_list[:, :, 15 * 3: 15 * 3 + 21 * 3]
right_hand_motion_sequence_list = full_body_motion_sequence_list[:, :, 15 * 3 + 21 * 3:]

with open(r'Z:\EgoMocap\work\EgocentricFullBody\dataset_files\diffusion_fullbody\ego_mean_std.pkl', 'rb') as f:
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
    if motion_i not in motion_id:
        continue

    left_hand_motion[:, 0] *= 0
    right_hand_motion[:, 0] *= 0

    assert len(left_hand_motion) == len(mo2cap2_motion) == len(right_hand_motion)

    if render:
        render_image_save_dir = os.path.join(save_dir, f'render_imgs_{motion_i}')
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


        left_hand_mesh = draw_skeleton_with_chain(left_hand_pose, mano_skeleton, keypoint_radius=0.01,
                                                  line_radius=0.0025)
        right_hand_mesh = draw_skeleton_with_chain(right_hand_pose, mano_skeleton, keypoint_radius=0.01,
                                                    line_radius=0.0025)
        mo2cap2_mesh = draw_skeleton_with_chain(mo2cap2_pose, mo2cap2_chain)
        coor = open3d.geometry.TriangleMesh.create_coordinate_frame()
        if only_hand:
            overall_mesh = left_hand_mesh
        else:
            overall_mesh = mo2cap2_mesh + left_hand_mesh + right_hand_mesh + coor



        if vis:

            open3d.visualization.draw_geometries([overall_mesh, coor])
        if save:
            mesh_save_path = os.path.join(save_dir, 'hand_%04d.ply' % i)
            open3d.io.write_triangle_mesh(mesh_save_path, overall_mesh)
        if save_viewpoint:
            save_view_point([overall_mesh, coor], view_point_save_path)
        if render:
            render_open3d([overall_mesh], view_point_save_path,
                          out_path=os.path.join(render_image_save_dir, 'hand_%04d.png' % i))

            # use ffmpeg to render video
            # ffmpeg -r 30 -i hand_%04d.png -vcodec libx264 -pix_fmt yuv420p -crf 25 -y hand.mp4
            import subprocess
            import shlex

    if render:
        subprocess.run(shlex.split(
            f'ffmpeg -r 30 -i {render_image_save_dir}/hand_%04d.png '
            f'-vcodec libx264 -pix_fmt yuv420p -y '
            f'{render_image_save_dir}/hand.mp4'))
