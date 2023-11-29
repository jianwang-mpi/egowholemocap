#  Copyright Jian Wang @ MPI-INF (c) 2023.
import copy
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from mmpose.datasets.builder import DATASETS

@DATASETS.register_module()
class StudioMotionDataset(Dataset):
    def __init__(self, path_dict):
        self.path_dict = path_dict



    def __len__(self):
        return len(self.data)

    def load_annotation(self):

        data = {}
        for seq_name in self.path_dict.keys():
            seq_data = self.path_dict[seq_name]

            motion_data_path = torch.load(seq_data['data_path'])
            lengths = torch.load(seq_data['length_path'])
            fps = seq_data['fps']
            max_frames = seq_data['max_frames']
            std_path = torch.load(seq_data['std_path'])
            mean_path = torch.load(seq_data['mean_path'])

            with open(motion_data_path, 'rb') as f:
                motion_data = pickle.load(f)

            # clip the motion data into a number of short sequences with length of max_frames
            motion_data = motion_data[:, :, :max_frames]





    def prepare_data(self, idx):
        """Get data sample."""
        result = self.data[idx]
        return result

    def __getitem__(self, idx):
        """Get a sample with given index."""
        # print(f'get item {idx}')
        results = copy.deepcopy(self.prepare_data(idx))
        return self.pipeline(results)


