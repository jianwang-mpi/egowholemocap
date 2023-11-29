#  Copyright Jian Wang @ MPI-INF (c) 2023.
#  Copyright Jian Wang @ MPI-INF (c) 2023.
import copy
import os
import pickle

import torch
from torch.utils.data import Dataset

from mmpose.datasets.builder import DATASETS
from mmpose.datasets.pipelines import Compose


@DATASETS.register_module()
class EgoSMPLXDataset(Dataset):

    def __init__(self,
                 data_path_list,
                 seq_len,
                 skip_frame,
                 pipeline,
                 target_frame_rate=25,
                 split_sequence=True,
                 test_mode=False):
        self.data_path_list = data_path_list  # smplx joint path list
        self.skip_frame = skip_frame
        self.seq_len = seq_len
        self.target_frame_rate = target_frame_rate

        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.data = self.load_annotation(split_sequence=split_sequence)

        self.eval_times = 0

    def load_annotation(self, split_sequence=True):
        data = []
        for data_item in self.data_path_list:
            data_path = data_item['path']
            data_frame_rate = data_item['frame_rate']
            assert data_frame_rate >= self.target_frame_rate
            frame_rate_skip = int(data_frame_rate / self.target_frame_rate)
            print('frame rate skip is {}'.format(frame_rate_skip))
            print('loading data from {}'.format(data_path))
            with open(data_path, 'rb') as f:
                data_i = pickle.load(f)
            smplx_joint_list = data_i['aligned_smplx_joints']
            for seq in smplx_joint_list:
                if split_sequence:
                    seq_list = self.split_into_sequences(seq, skip=self.skip_frame * frame_rate_skip,
                                                         frame_skip=frame_rate_skip)
                else:
                    seq_list = [seq[::frame_rate_skip]]
                data.extend(seq_list)

        return data

    def split_into_sequences(self, seq, skip=10, frame_skip=1):
        seq_len = self.seq_len
        result = []
        for i in range(0, len(seq) - seq_len + 1, skip):
            result.append(seq[i: i + seq_len: frame_skip])
        return result

    def __len__(self):
        return len(self.data)

    def prepare_data(self, idx):
        data_i = self.data[idx]
        data_i = torch.as_tensor(data_i, dtype=torch.float32)
        result = {'aligned_smplx_joints': data_i}
        return result

    def evaluate(self, outputs, res_folder, metric=None, logger=None):
        # just save the outputs
        save_path = os.path.join(res_folder, f'outputs_eval_{self.eval_times}.pkl') # stupid way to record the epoch id
        with open(save_path, 'wb') as f:
            pickle.dump(outputs, f)
        self.eval_times += 1
        evaluation_results = {'mpjpe': 0}
        return evaluation_results

    def __getitem__(self, idx):
        """Get a sample with given index."""
        # print(f'get item {idx}')
        results = copy.deepcopy(self.prepare_data(idx))
        return self.pipeline(results)