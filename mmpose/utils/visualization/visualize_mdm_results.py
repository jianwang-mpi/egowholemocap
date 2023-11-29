#  Copyright Jian Wang @ MPI-INF (c) 2023.

import os

from mmpose.models.diffusion_mdm.data_loaders.humanml.utils import paramUtil
from mmpose.models.diffusion_mdm.data_loaders.humanml_utils import MO2CAP2_TREE_IN_HUMANML
from mmpose.models.diffusion_mdm.data_loaders.humanml.utils.plot_script import plot_3d_motion


def render_mdm_results(input_motions, result_motions, save_dir):
    """
    Render the input and output motions from MDM.
    :param input_motions: (N, T, joint_num, 3)
    :param result_motions: (N, T, joint_num, 3)
    :return:
    """
    assert input_motions.shape == result_motions.shape
    assert input_motions.shape[2] == 22
    assert input_motions.shape[3] == 3

    batch_size, T, joint_num, _ = input_motions.shape



    for sample_i in range(batch_size):
        input_caption = 'Input Motion'
        save_file = 'input_motion{:02d}.mp4'.format(sample_i)
        animation_save_path = os.path.join(save_dir, save_file)
        rep_files = [animation_save_path]
        print(f'[({sample_i}) "{input_caption}" | -> {save_file}]')
        input_motion = input_motions[sample_i]

        plot_3d_motion(animation_save_path, MO2CAP2_TREE_IN_HUMANML, input_motion, title=input_caption,
                       dataset='humanml', fps=20, vis_mode='gt',
                       gt_frames=[])

        # draw edited results
        edited_motion_caption = 'edit result'
        length = result_motions.shape[1]
        save_file = 'edit_result{:02d}.mp4'.format(sample_i)
        animation_result_save_path = os.path.join(save_dir, save_file)
        rep_files.append(animation_result_save_path)
        result_motion = result_motions[sample_i]
        print(f'[({sample_i}) "{edited_motion_caption}"  | -> {save_file}]')

        plot_3d_motion(animation_result_save_path, MO2CAP2_TREE_IN_HUMANML, result_motion, title=edited_motion_caption,
                       dataset='humanml', fps=20, vis_mode='default',
                       gt_frames=[])

        all_rep_save_file = os.path.join(save_dir, 'sample{:02d}.mp4'.format(sample_i))
        ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
        hstack_args = f' -filter_complex hstack=inputs={2}'
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
        os.system(ffmpeg_rep_cmd)
        print(f'[({sample_i}) | all repetitions | -> {all_rep_save_file}]')

