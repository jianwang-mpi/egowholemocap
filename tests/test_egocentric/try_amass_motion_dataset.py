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


def try_amass_dataset():

    pipeline = [
        Collect(keys=['data', 'mask', 'lengths'],
                meta_keys=[])
    ]

    path_dict = {
        'seq1': r'Z:\datasets00\static00\AMASS\AMASS_SMPLX_G\CMU\01\01_03_stageii.npz'
    }

    dataset_cfg = dict(
        type='AMASSMotionDataset',
        path_dict=path_dict,
        frame_rate=20,
        seq_len=196,
        std_path=r'Z:\EgoMocap\work\motion-diffusion-model\dataset\HumanML3D\Std.npy',
        mean_path=r'Z:\EgoMocap\work\motion-diffusion-model\dataset\HumanML3D\Mean.npy',
        smplx_model_dir=r'Z:\EgoMocap\work\EgocentricFullBody\human_models\smplx_new',
        pipeline=pipeline,
        test_mode=True)

    amass_dataset = build_dataset(dataset_cfg)

    print(f'length of dataset is: {len(amass_dataset)}')

    seq_id = 0
    data_i = amass_dataset[seq_id]

    seq_i = data_i['data']
    # convert to bgr and save with opencv
    print(seq_i.shape)
    length_i = data_i['lengths']
    mask_i = data_i['mask']
    print(length_i)
    print(mask_i)
    print(mask_i.shape)

    # visualize the dataset with open3d

    from mmpose.datasets.datasets.diffusion.keypoints_to_hml3d import recover_from_ric
    from mmpose.utils.visualization.draw import draw_skeleton_with_chain
    from mmpose.models.diffusion_mdm.data_loaders.humanml.utils import paramUtil
    joint_chain = paramUtil.t2m_kinematic_chain
    if isinstance(seq_i, np.ndarray):
        seq_i = torch.asarray(seq_i)
    seq_i = seq_i * amass_dataset.std + amass_dataset.mean
    joint_location_seq = recover_from_ric(seq_i, 22)
    if not isinstance(joint_location_seq, np.ndarray):
        joint_location_seq = joint_location_seq.numpy()
    coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
    for pose_i in joint_location_seq:
        skeleton_i = draw_skeleton_with_chain(pose_i, joint_chain)
        open3d.visualization.draw_geometries([skeleton_i, coord])


if __name__ == '__main__':
    try_amass_dataset()
