import os

import cv2
import numpy as np
import open3d
import torch

from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import (LoadImageFromFile, CropCircle, Generate2DPose,
                                       CropImage, ResizeImage, Generate2DPoseConfidence, ToTensor, NormalizeTensor,
                                       TopDownGenerateTarget, Collect)
from mmpose.models.diffusion_mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
from mmpose.utils.visualization.draw import draw_joints
from mmpose.utils.visualization.skeleton import Skeleton

if os.name == 'nt':
    seq_path = r'Z:/datasets00/static00/AMASS/AMASS_SMPLX_G/CMU/01/01_03_stageii.npz'
    std_path = r'Z:/EgoMocap/work/motion-diffusion-model\dataset\HumanML3D\Std.npy'
    mean_path = r'Z:\EgoMocap\work\motion-diffusion-model\dataset\HumanML3D\Mean.npy'
    smplx_model_dir = r'Z:\EgoMocap\work\EgocentricFullBody\human_models\smplx_new'
    mdm_model_path = r'Z:\EgoMocap\work\motion-diffusion-model\save\humanml_trans_enc_512\model000200000.pt'
    diffusion_video_save_dir = r'Z:\EgoMocap\work\EgocentricFullBody\vis_results\diffusion_results'
else:
    seq_path = r'/CT/datasets00/static00/AMASS/AMASS_SMPLX_G/CMU/01/01_03_stageii.npz'
    std_path = r'/CT/EgoMocap/work/motion-diffusion-model/dataset/HumanML3D/Std.npy'
    mean_path = r'/CT/EgoMocap/work/motion-diffusion-model/dataset/HumanML3D/Mean.npy'
    smplx_model_dir = r'/CT/EgoMocap/work/EgocentricFullBody/human_models/smplx_new'
    mdm_model_path = r'/CT/EgoMocap/work/motion-diffusion-model/save/humanml_trans_enc_512/model000200000.pt'
    diffusion_video_save_dir = r'/CT/EgoMocap/work/EgocentricFullBody/vis_results/diffusion_results'

def try_amass_dataset(data_id=0, vis=False):



    pipeline = [
        Collect(keys=['data', 'mask', 'lengths', 'mean', 'std'],
                meta_keys=[])
    ]

    path_dict = {
        'seq1': seq_path
    }

    dataset_cfg = dict(
        type='AMASSMotionDataset',
        path_dict=path_dict,
        frame_rate=20,
        seq_len=196,
        std_path=std_path,
        mean_path=mean_path,
        smplx_model_dir=smplx_model_dir,
        pipeline=pipeline,
        test_mode=True)

    amass_dataset = build_dataset(dataset_cfg)

    print(f'length of dataset is: {len(amass_dataset)}')

    data_i = amass_dataset[data_id]


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

    data_i['motions'] = joint_location_seq

    if vis:
        coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
        for pose_i in joint_location_seq:
            skeleton_i = draw_skeleton_with_chain(pose_i, joint_chain)
            open3d.visualization.draw_geometries([skeleton_i, coord])

    return data_i


if __name__ == '__main__':
    data_i = try_amass_dataset(vis=False)
    for k, v in data_i.items():
        if k in ['data', 'mask', 'lengths', 'mean', 'std', 'motions']:
            data_i[k] = torch.asarray(v).unsqueeze(0).cuda().float()

    mean = data_i['mean']
    std = data_i['std']

    from mmpose.models.diffusion_mdm.edit import MDMEdit
    mdm_edit = MDMEdit(model_path=mdm_model_path,
                       max_frames=196).cuda()
    mdm_edit.eval()
    with torch.no_grad():
        all_motions = mdm_edit(**data_i)

    out_motions = all_motions['motions']
    input_motions = all_motions['input_motions']
    input_motions = input_motions.permute(0, 2, 3, 1)
    input_motions = input_motions * std + mean  # this should be in the dataset!
    input_motions = recover_from_ric(input_motions, 22)
    # sample shape: (batch_size, 1, max_frames, 22, 3)
    input_motions = input_motions.view(-1, *input_motions.shape[2:])
    input_motions = input_motions.cpu().numpy()

    print(f'save at: {diffusion_video_save_dir}')
    os.makedirs(diffusion_video_save_dir, exist_ok=True)

    mdm_edit.show_result(results=out_motions, inputs=input_motions, save_dir=diffusion_video_save_dir)
    # pose_example = all_motions[0]
    # from mmpose.utils.visualization.draw import draw_skeleton_with_chain
    # from mmpose.models.diffusion_mdm.data_loaders.humanml.utils import paramUtil
    #
    # joint_chain = paramUtil.t2m_kinematic_chain
    # coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
    # for pose_i in pose_example:
    #     skeleton_i = draw_skeleton_with_chain(pose_i, joint_chain)
    #     open3d.visualization.draw_geometries([skeleton_i, coord])

