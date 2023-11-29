#  Copyright Jian Wang @ MPI-INF (c) 2023.
import open3d

import torch

import numpy as np

from mmpose.models.diffusion_mdm.data_loaders.humanml.utils import paramUtil
from mmpose.utils.visualization.draw import draw_skeleton_with_chain


def main():
    skeleton_chain = paramUtil.t2m_kinematic_chain

    output_file_path = r'Z:\EgoMocap\work\EgocentricHand\motion-diffusion-model\dataset\HumanML3D\new_joints' \
                       r'\012314.npy'

    motion_data = np.load(output_file_path, allow_pickle=True)

    print(motion_data.shape)
    coord = open3d.geometry.TriangleMesh.create_coordinate_frame()

    for i in range(motion_data.shape[0]):
        pose = motion_data[i]
        skeleton_mesh = draw_skeleton_with_chain(pose, skeleton_chain)
        open3d.visualization.draw_geometries([skeleton_mesh, coord])

if __name__ == '__main__':
    main()