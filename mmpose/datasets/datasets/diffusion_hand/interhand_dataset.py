#  Copyright Jian Wang @ MPI-INF (c) 2023.
import copy
import json
import os

import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset

from mmpose.datasets.builder import DATASETS
from mmpose.datasets.pipelines import Compose


@DATASETS.register_module()
class InterHandDataset(Dataset):
    frame_rate = 30

    def __init__(self, data_path,
                 seq_len,
                 skip_frames,
                 pipeline, split_sequence=True, use_split='train', test_mode=False):
        self.data_path = data_path
        self.use_split = use_split

        self.seq_len = seq_len
        self.skip_frames = skip_frames

        self.img_root_path = os.path.join(self.data_path, 'images')
        self.annot_root_path = os.path.join(self.data_path, 'annotations')

        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        self.data = self.load_annotation(split_sequence=split_sequence)

        self.epoch_num = 0

    def load_annotation(self, split_sequence=False):
        # do not use images here
        with open(os.path.join(self.annot_root_path, self.use_split,
                               'InterHand2.6M_' + self.use_split + '_joint_3d.json')) as f:
            joints = json.load(f)
        capture_index_list = natsorted(list(joints.keys()))
        # split left hand and right hand
        # todo: split left hand and right hand
        left_hand_joint_seq_list = []
        for capture_index in capture_index_list:
            frame_index_list = natsorted(list(joints[capture_index].keys()))
            joint_seq = []
            for frame_index in frame_index_list:
                joint_hand = np.array(joints[capture_index][frame_index]['world_coord'], dtype=np.float32).reshape(-1,
                                                                                                                   3)
                joint_seq.append(joint_hand)
            if split_sequence:
                joint_seqs = self.split_into_sequences(joint_seq)
            else:
                joint_seqs = [joint_seq]
            joint_seq_list.extend(joint_seqs)

        return joint_seq_list

    def split_into_sequences(self, seq):
        seq_len = self.seq_len
        skip_frames = self.skip_frames
        result = []
        for i in range(0, len(seq) - seq_len + 1, skip_frames):
            result.append(seq[i:i + seq_len])
        return result

    def __len__(self):
        return len(self.data)

    def prepare_data(self, idx):
        """Get data sample."""
        joint_seq = self.data[idx]
        result_dict = {}
        result_dict['global_hand_joints'] = joint_seq
        return result_dict

    def __getitem__(self, idx):
        """Get a sample with given index."""
        # print(f'get item {idx}')
        results = copy.deepcopy(self.prepare_data(idx))
        return self.pipeline(results)
