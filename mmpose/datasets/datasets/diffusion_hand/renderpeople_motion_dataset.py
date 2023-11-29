#  Copyright Jian Wang @ MPI-INF (c) 2023.
#  Copyright Jian Wang @ MPI-INF (c) 2023.
import copy
import json
import pickle

import torch
import numpy as np
import cv2
import os
import smplx
from natsort import natsorted
from tqdm import tqdm

from mmpose.datasets.builder import DATASETS
from torch.utils.data import Dataset
from mmpose.datasets.pipelines import Compose
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model

from mmpose.datasets.datasets.diffusion_hand.smplx_forward import SMPLXForward
@DATASETS.register_module()
class RenderpeopleMotionDataset(Dataset):
    frame_rate = 30
    def __init__(self, data_path,
                 seq_len,
                 skip_frames,
                 pipeline, split_sequence=True,
                 human_names=None, test_mode=False):
        self.data_path = data_path

        self.seq_len = seq_len
        self.skip_frames = skip_frames

        if human_names is None:
            self.data_dirs = ['render_people_claudia',
                              'render_people_eric',
                              'render_people_carla',
                              'render_people_adanna',
                              'render_people_amit',
                              'render_people_janna',
                              'render_people_joko',
                              'render_people_joyce',
                              'render_people_kyle',
                              'render_people_maya',
                              'render_people_rin',
                              'render_people_scott',
                              'render_people_serena',
                              'render_people_shawn',
                              ]
        else:
            self.data_dirs = human_names

        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        self.data = self.load_annotation(split_sequence=split_sequence)

        self.epoch_num = 0

        self.renderpeople_index, self.smplx_index = dset_to_body_model(dset='renderpeople', model_type='smplx')

    def load_annotation(self, split_sequence=True):
        data_seq = []
        for human_name in self.data_dirs:
            pkl_path = os.path.join(self.data_path, f'{human_name}.pkl')

            print('loading data from {}'.format(pkl_path))
            with open(pkl_path, 'rb') as f:
                identity_data = pickle.load(f)
            for seq_name, seq_data in tqdm(identity_data.items()):
                if split_sequence:
                    seq_data = self.split_into_sequences(seq_data)
                    if len(seq_data) > 0:
                        data_seq.extend(seq_data)
                    else:
                        print(f'seq {seq_name} has no enough data')
                else:
                    if len(seq_data) > self.seq_len:
                        data_seq.append(seq_data)
                    else:
                        print(f'seq {seq_name} has no enough data')
        return data_seq


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
        sequence = self.data[idx]

        # convert from renderpeople joints to smplx joints under the egocentric coordinate system
        ego_smplx_joint_list = np.zeros((len(sequence), 145, 3))
        renderpeople_ego_joint_list = [item['renderpeople_local_joints'] for item in sequence]
        renderpeople_ego_joint_list = np.asarray(renderpeople_ego_joint_list)
        ego_smplx_joint_list[:, self.smplx_index] = renderpeople_ego_joint_list[:, self.renderpeople_index]

        # convert from renderpeople joints to smplx joints under the world coordinate system
        global_smplx_joint_list = np.zeros((len(sequence), 145, 3))
        renderpeople_global_joint_list = [item['renderpeople_joints'] for item in sequence]
        renderpeople_global_joint_list = np.asarray(renderpeople_global_joint_list)
        global_smplx_joint_list[:, self.smplx_index] = renderpeople_global_joint_list[:, self.renderpeople_index]
        # global smplx joints in z upwards, now we need it to be y upwards
        trans_matrix = np.array([[-1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0],
                                 [0.0, 1.0, 0.0]])
        global_smplx_joint_list = np.dot(global_smplx_joint_list, trans_matrix)

        result_dict = {
            'ego_smplx_joints': ego_smplx_joint_list,
            'global_smplx_joints': global_smplx_joint_list,
        }

        return result_dict

    def evaluate(self, outputs, res_folder, metric=None, logger=None):
        # just save the outputs
        save_path = os.path.join(res_folder, f'outputs_{self.epoch_num}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(outputs, f)
        self.epoch_num += 1
        evaluation_results = {'mpjpe': 0}
        return evaluation_results

    def __getitem__(self, idx):
        """Get a sample with given index."""
        # print(f'get item {idx}')
        results = copy.deepcopy(self.prepare_data(idx))
        return self.pipeline(results)
