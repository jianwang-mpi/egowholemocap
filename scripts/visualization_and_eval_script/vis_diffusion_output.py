#  Copyright Jian Wang @ MPI-INF (c) 2023.
import open3d

import torch

import numpy as np

from mmpose.models.diffusion_mdm.data_loaders.humanml.utils import paramUtil
from mmpose.utils.visualization.draw import draw_skeleton_with_chain


def main():
    transform_left_hand_to_right_hand = np.asarray([[1, 0, 0],
                                                   [0, 0, 1],
                                                   [0, 1, 0]])
    skeleton_chain = paramUtil.kit_kinematic_chain

    output_file_path = r'Z:\EgoMocap\work\EgocentricHand\motion-diffusion-model\save\kit_trans_enc_512' \
                       r'\edit_kit_trans_enc_512_000400000_in_between_seed10\results.npy'

    data = np.load(output_file_path, allow_pickle=True)
    data = data.reshape(1)
    data = data[0]

    motion_data = data['motion'] * 0.003
    print(motion_data.shape)
    coord = open3d.geometry.TriangleMesh.create_coordinate_frame()

    for i in range(motion_data.shape[0]):
        motion = motion_data[i]
        motion = np.transpose(motion, (2, 0, 1))
        motion = np.ascontiguousarray(motion, dtype=np.float32)
        for pose in motion:
            # pose from left hand coord to right hand coord
            # pose = pose @ transform_left_hand_to_right_hand
            skeleton_mesh = draw_skeleton_with_chain(pose, skeleton_chain)
            open3d.visualization.draw_geometries([skeleton_mesh, coord])

if __name__ == '__main__':
    main()