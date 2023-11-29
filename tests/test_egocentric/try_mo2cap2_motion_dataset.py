import os

import cv2
import numpy as np
import open3d
import torch

from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import (LoadImageFromFile, CropCircle, Generate2DPose,
                                       CropImage, ResizeImage, Generate2DPoseConfidence, ToTensor, NormalizeTensor,
                                       TopDownGenerateTarget, Collect)
from mmpose.utils.visualization.draw import draw_joints
from mmpose.utils.visualization.skeleton import Skeleton


if os.name == 'nt':
    # windows
    # seq_path = r'X:\ScanNet\work\egocentric_view\25082022\jian1\pose_gt.pkl'
    seq_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\vit_256x256_heatmap_3d_test\results_global.pkl'
    std_path = r'Z:/EgoMocap/work/motion-diffusion-model\dataset\HumanML3D\Std.npy'
    mean_path = r'Z:\EgoMocap\work\motion-diffusion-model\dataset\HumanML3D\Mean.npy'
else:
    # seq_path = r'/HPS/ScanNet/work/egocentric_view/25082022/jian1/pose_gt.pkl'
    seq_path = r'/CT/EgoMocap/work/EgocentricFullBody/work_dirs/vit_256x256_heatmap_3d_test/results_global.pkl'
    std_path = r'/CT/EgoMocap/work/motion-diffusion-model/dataset/HumanML3D/Std.npy'
    mean_path = r'/CT/EgoMocap/work/motion-diffusion-model/dataset/HumanML3D/Mean.npy'

def try_amass_dataset():

    pipeline = [
        Collect(keys=['data', 'mask', 'lengths'],
                meta_keys=[])
    ]

    # path_dict = {
    #     'seq1': seq_path
    # }
    path = seq_path
    dataset_cfg = dict(
        type='Mo2Cap2MotionDataset',
        path_dict=path,
        frame_rate=20,
        seq_len=196,
        std_path=std_path,
        mean_path=mean_path,
        pipeline=pipeline,
        test_mode=True)

    mo2cap2_dataset = build_dataset(dataset_cfg)

    print(f'length of dataset is: {len(mo2cap2_dataset)}')

    seq_id = 0
    data_i = mo2cap2_dataset[seq_id]

    seq_i = data_i['data']

    # visualize the dataset with open3d

    from mmpose.datasets.datasets.diffusion.keypoints_to_hml3d import recover_from_ric
    from mmpose.utils.visualization.draw import draw_skeleton_with_chain
    from mmpose.models.diffusion_mdm.data_loaders.humanml.utils import paramUtil

    if isinstance(seq_i, np.ndarray):
        seq_i = torch.asarray(seq_i)
    seq_i = seq_i * mo2cap2_dataset.std + mo2cap2_dataset.mean
    joint_location_seq = recover_from_ric(seq_i, 22)
    if not isinstance(joint_location_seq, np.ndarray):
        joint_location_seq = joint_location_seq.numpy()
    coord = open3d.geometry.TriangleMesh.create_coordinate_frame()

    # from human ml to mo2cap2
    mo2cap2_pose = np.zeros((len(joint_location_seq), 15, 3))
    mo2cap2_pose[:, mo2cap2_dataset.dst_idxs] = joint_location_seq[:, mo2cap2_dataset.model_idxs]
    joint_chain = [[0, 1, 2, 3], [0, 4, 5, 6], [1, 7, 8, 9, 10], [4, 11, 12, 13, 14], [7, 11]]
    for pose_i in mo2cap2_pose:

        skeleton_i = draw_skeleton_with_chain(pose_i, joint_chain)
        open3d.visualization.draw_geometries([skeleton_i, coord])


if __name__ == '__main__':
    try_amass_dataset()
