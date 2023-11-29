#  Copyright Jian Wang @ MPI-INF (c) 2023.
#  Copyright Jian Wang @ MPI-INF (c) 2023.
import copy
import json
import pickle
from scripts.captury_studio_tools.npybvh.bvh import Bvh
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


@DATASETS.register_module()
class BEATDataset(Dataset):
    frame_rate = 120

    def __init__(self, data_path,
                 seq_len,
                 skip_frames,
                 pipeline,
                 output_frame_rate=30,
                 split_sequence=True,
                 data_ids=None,
                 test_mode=False):
        self.data_path = data_path
        self.output_frame_rate = output_frame_rate

        self.seq_len = seq_len
        self.skip_frames = skip_frames

        if data_ids is None:
            self.data_ids = np.arange(1, 31, 1)
        else:
            self.data_ids = data_ids

        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        self.data = self.load_annotation(split_sequence=split_sequence)

        self.epoch_num = 0

        self.beat_index, self.smplx_index = dset_to_body_model(dset='beat', model_type='smplx')

    def parse_bvh_file(self, bvh_file_path, start_frame=0, input_frame_rate=120, output_frame_rate=30,
                       joint_position_scale=1 / 100):
        anim = Bvh()
        anim.parse_file(bvh_file_path)
        gt_pose_seq = []
        # print(anim.frames)
        # print(anim.joint_names())
        joint_name_list = list(anim.joint_names())
        # egocentric_joints = [joint_name_list.index(jt_name) for jt_name in studio_original_joint_names]
        step = round(input_frame_rate / output_frame_rate)
        for frame in tqdm(range(start_frame, anim.frames, step)):
            positions, rotations = anim.frame_pose(frame)

            # positions = positions[egocentric_joints]
            positions = positions * joint_position_scale
            gt_pose_seq.append(positions)

        gt_pose_seq = np.asarray(gt_pose_seq)
        return gt_pose_seq

    def load_annotation(self, split_sequence=True):
        data_seq = []
        for data_id in self.data_ids:
            data_id_path = os.path.join(self.data_path, str(data_id))
            data_id_dir_list = natsorted(os.listdir(data_id_path))
            data_id_dir_list = [data_file for data_file in data_id_dir_list if data_file.endswith('bvh')]
            print(data_id_dir_list)
            for data_file in data_id_dir_list:

                seq_data = self.parse_bvh_file(os.path.join(data_id_path, data_file), start_frame=0,
                                               input_frame_rate=self.frame_rate,
                                               output_frame_rate=self.output_frame_rate,
                                               joint_position_scale=1 / 100)
                if split_sequence:
                    seq_data = self.split_into_sequences(seq_data)
                    if len(seq_data) > 0:
                        data_seq.extend(seq_data)
                    else:
                        print(f'seq {data_file} has no enough data')
                else:
                    data_seq.append(seq_data)
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


        # convert from renderpeople joints to smplx joints under the world coordinate system
        global_smplx_joint_list = np.zeros((len(sequence), 145, 3))
        global_smplx_joint_list[:, self.smplx_index] = sequence[:, self.renderpeople_index]
        # global smplx joints in z upwards, now we need it to be y upwards
        trans_matrix = np.array([[-1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0],
                                 [0.0, 1.0, 0.0]])
        global_smplx_joint_list = np.dot(global_smplx_joint_list, trans_matrix)

        result_dict = {
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
