#  Copyright Jian Wang @ MPI-INF (c) 2023.
import copy
import os.path
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from mmpose.datasets.builder import DATASETS
import smplx
from mmpose.datasets.datasets.diffusion.keypoints_to_hml3d import KeypointsToHumanML3D
from mmpose.datasets.pipelines import Compose


@DATASETS.register_module()
class AMASSMotionDataset(Dataset):
    def __init__(self, path_dict,
                 frame_rate,
                 seq_len,
                 mean_path,
                 std_path,
                 smplx_model_dir,
                 pipeline,
                 test_mode=False):
        self.path_dict = path_dict
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        smplx_male_model_path = os.path.join(smplx_model_dir, 'SMPLX_MALE.npz')
        self.male_smplx = smplx.create(smplx_male_model_path, model_type='smplx', num_betas=10, use_pca=False,
                                       gender='male').cuda()
        smplx_female_model_path = os.path.join(smplx_model_dir, 'SMPLX_FEMALE.npz')
        self.female_smplx = smplx.create(smplx_female_model_path, model_type='smplx', num_betas=10, use_pca=False,
                                         gender='female').cuda()

        self.output_frame_rate = frame_rate
        self.seq_len = seq_len
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

        self.keypoints_seq = self.load_annotation()

        self.keypoints_to_hml3d_class = KeypointsToHumanML3D()
        self.data = self.keypoints_to_hml3d(self.keypoints_seq)
        self.data = self.split_into_sequences(self.data)
        self.data = self.normalize_data(self.data)

    def normalize_data(self, data):
        for i in range(len(data)):
            data[i]['data'] = (data[i]['data'] - self.mean) / self.std
        return data

    def __len__(self):
        return len(self.data)

    def keypoints_to_hml3d(self, keypoints_seq_list):
        hml3d_seq_list = []
        for keypoints_seq in keypoints_seq_list:
            data, ground_positions, positions, l_velocity = self.keypoints_to_hml3d_class.process_file(
                keypoints_seq, 0.002)

            hml3d_seq_list.append(data)
        return hml3d_seq_list

    def load_annotation(self):
        trans_matrix = np.array([[1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0],
                                 [0.0, 1.0, 0.0]])
        joint_location_seqs = []
        for seq_name in self.path_dict.keys():
            motion_data_path = self.path_dict[seq_name]

            motion_sequence = np.load(motion_data_path)

            # clip the motion data into a number of short sequences with length of max_frames
            seq_length = len(motion_sequence['poses'])
            subject_gender = motion_sequence['gender']
            fps = motion_sequence['mocap_frame_rate']
            downsample = int(fps / self.output_frame_rate)

            keypoints_seq = []
            with torch.no_grad():
                for i in range(0, seq_length, downsample):
                    body_parms = {
                        'global_orient': torch.Tensor(motion_sequence['poses'][i:i + 1, :3]),
                        # controls the global root orientation
                        'body_pose': torch.Tensor(motion_sequence['poses'][i:i + 1, 3:66]),
                        # controls the body
                        'transl': torch.Tensor(motion_sequence['trans'][i:i + 1]),
                        # controls the global body position
                        'betas': torch.Tensor(motion_sequence['betas'][:10][np.newaxis])
                    }

                    for k in body_parms.keys():
                        body_parms[k] = body_parms[k].cuda()

                    if np.array_str(subject_gender).lower() == 'male':
                        body_model_out = self.male_smplx(**body_parms)
                    else:
                        body_model_out = self.female_smplx(**body_parms)

                    joint_locations = body_model_out.joints.detach().cpu().numpy()
                    joint_locations = joint_locations[:, :22]
                    keypoints_seq.append(joint_locations)
            keypoints_seq = np.concatenate(keypoints_seq, axis=0)
            pose_seq_np_n = np.dot(keypoints_seq, trans_matrix)
            pose_seq_np_n[..., 0] *= -1
            # print(pose_seq_np_n[0][17])
            # print(pose_seq_np_n[0][16])
            #
            # import open3d
            # from mmpose.utils.visualization.draw import draw_skeleton_with_chain, get_arrow
            # print('visualize pose...')
            # from mmpose.models.diffusion_mdm.data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
            # skeleton_mesh = draw_skeleton_with_chain(pose_seq_np_n[0], t2m_kinematic_chain)
            # coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
            # open3d.visualization.draw_geometries([skeleton_mesh, coord])

            joint_location_seqs.append(pose_seq_np_n)
        return joint_location_seqs

    def split_into_sequences(self, seq_list):
        result_list = []
        for seq in seq_list:
            seq_length = len(seq)
            for i in range(0, seq_length, self.seq_len):
                result_seq = seq[i:i + self.seq_len]
                mask = np.ones((result_seq.shape[0], result_seq.shape[1], 1))
                current_len_result_seq = len(result_seq)
                if current_len_result_seq < self.seq_len:
                    result_seq = np.concatenate([result_seq,
                                     np.zeros((self.seq_len - current_len_result_seq, result_seq.shape[1]))
                                     ], axis=0)
                    mask = np.concatenate([mask,
                                             np.zeros((self.seq_len - current_len_result_seq, result_seq.shape[1], 1))
                                             ], axis=0)
                result_list.append({'data': result_seq, 'mask': mask, 'lengths': np.asarray(current_len_result_seq),
                                    'mean': self.mean, 'std': self.std})
        return result_list

    def prepare_data(self, idx):
        """Get data sample."""
        result = self.data[idx]
        return result

    def __getitem__(self, idx):
        """Get a sample with given index."""
        # print(f'get item {idx}')
        results = copy.deepcopy(self.prepare_data(idx))
        return self.pipeline(results)
