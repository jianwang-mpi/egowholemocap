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
class MDMExampleDataset(Dataset):
    def __init__(self, path_dict,
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

        self.seq_len = seq_len
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

        self.keypoints_seq = self.load_annotation()

        self.keypoints_seq = self.split_into_sequences(self.keypoints_seq)
        self.data = self.normalize_data(self.keypoints_seq)

    def normalize_data(self, data):
        for i in range(len(data)):
            data[i]['data'] = (data[i]['data'] - self.mean) / self.std
        return data

    def __len__(self):
        return len(self.data)


    def load_annotation(self):
        joint_location_seqs = []
        for seq_name in self.path_dict.keys():
            motion_data_path = self.path_dict[seq_name]
            motion_sequence = np.load(motion_data_path)
            joint_location_seqs.append(motion_sequence)

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
