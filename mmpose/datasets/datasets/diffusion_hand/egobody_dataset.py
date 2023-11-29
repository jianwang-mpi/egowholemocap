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

from mmpose.datasets.builder import DATASETS
from torch.utils.data import Dataset
from mmpose.datasets.pipelines import Compose

from mmpose.datasets.datasets.diffusion_hand.smplx_forward import SMPLXForward
@DATASETS.register_module()
class EgoBodyDataset(Dataset):
    frame_rate = 30
    def __init__(self, data_path,
                 seq_len,
                 skip_frames,
                 smplx_model_dir,
                 pipeline,
                 place_on_floor=False,
                 split_sequence=True,
                 data_dirs=None,
                 test_mode=False
                 ):
        self.data_path = data_path

        self.seq_len = seq_len
        self.skip_frames = skip_frames
        self.place_on_floor = place_on_floor
        # if place_on_floor is true, then place the human motion on floor

        if data_dirs is None:
            self.data_dirs = ['smplx_interactee_val', 'smplx_interactee_test', 'smplx_interactee_train',
                              'smplx_camera_wearer_val', 'smplx_camera_wearer_test', 'smplx_camera_wearer_train']
        else:
            self.data_dirs = data_dirs

        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        self.data, self.data_calib = self.load_annotation(split_sequence=split_sequence)

        self.smplx_forward = SMPLXForward(smplx_model_dir, use_pca=True, num_pca_comps=12)

        self.epoch_num = 0

    def load_annotation(self, split_sequence=True):
        calibration_dir = os.path.join(self.data_path, 'calibrations')

        data_seq = []
        data_seq_calib = []
        for data_dir in self.data_dirs:
            data_path = os.path.join(self.data_path, data_dir)
            print('loading data from {}'.format(data_path))
            recording_list = natsorted(os.listdir(data_path))
            for recording_name in recording_list:
                recording_path = os.path.join(data_path, recording_name)
                recording_calibration_path = os.path.join(calibration_dir, recording_name)
                for identity_name in natsorted(os.listdir(recording_path)):
                    seq_dir = os.path.join(recording_path, identity_name)
                    seq_calibration_dir = os.path.join(recording_calibration_path, 'cal_trans')
                    seq_list, seq_calibration_list = self.load_sequence(seq_dir, seq_calibration_dir, split_sequence)
                    data_seq.extend(seq_list)
                    data_seq_calib.extend(seq_calibration_list)
        return data_seq, data_seq_calib

    def load_sequence(self, seq_dir, seq_calibration_dir, split=True):
        seq_dir = os.path.join(seq_dir, 'results')
        seq_calibration_json_path = os.path.join(seq_calibration_dir, 'kinect12_to_world')
        seq_calibration_json_name_list = os.listdir(seq_calibration_json_path)
        assert len(seq_calibration_json_name_list) == 1
        seq_calibration_json_path = os.path.join(seq_calibration_json_path, seq_calibration_json_name_list[0])
        print(seq_calibration_json_path)
        seq = []
        for frame_name in natsorted(os.listdir(seq_dir)):
            frame_path = os.path.join(seq_dir, frame_name)
            frame_calibration_path = os.path.join(seq_calibration_dir, frame_name)
            pkl_path = os.path.join(frame_path, '000.pkl')
            seq.append(pkl_path)
        # split the sequence by seq_len
        if split:
            seq_list = self.split_into_sequences(seq)
        else:
            seq_list = [seq]
        return seq_list, [seq_calibration_json_path] * len(seq_list)

    def split_into_sequences(self, seq):
        seq_len = self.seq_len
        skip_frames = self.skip_frames
        result = []
        for i in range(0, len(seq) - seq_len + 1, skip_frames):
            result.append(seq[i:i + seq_len])
        return result

    def __len__(self):
        return len(self.data)


    def transform_3d_points_to_global_coord(self, points_3d, transformation_matrix):
        # assert points shape: (len, 3)
        seq_len, points_num, _ = points_3d.shape
        assert _ == 3
        # convert to homogenious coordinates
        points_homo = torch.concatenate([points_3d, torch.ones((seq_len, points_num, 1))], dim=-1)
        points_homo = points_homo.reshape(seq_len * points_num, 4)
        transformed_points = (transformation_matrix @ points_homo.T).T
        transformed_points = transformed_points.reshape(seq_len, points_num, 4)
        transformed_points = transformed_points[:, :, :3]
        return transformed_points

    def get_fisheye_camera_coord(self, global_smplx_vertices):
        pass

    def prepare_data(self, idx):
        """Get data sample."""
        pkl_path_seq = self.data[idx]
        json_path_calib = self.data_calib[idx]
        smplx_input_list = []
        gender = 'neutral'
        for pkl_path in pkl_path_seq:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            data_torch = {k: torch.from_numpy(v) for k, v in data.items() if k != 'gender'}
            gender = data['gender']
            smplx_input_list.append(data_torch)
        # convert to batch
        smplx_input_batch = {}
        for k in smplx_input_list[0].keys():
            smplx_input_batch[k] = torch.concatenate([data[k] for data in smplx_input_list], dim=0)
        # smplx_input_batch['betas'] = smplx_input_batch['betas'][0]  # only use the first beta
        smplx_result_dict = {'gender': gender}
        smplx_result_dict.update(smplx_input_batch)
        result_dict = {'smplx_input': smplx_result_dict}
        result_dict = self.smplx_forward(result_dict)

        # do transformation to smplx results
        smplx_joints = result_dict['smplx_output'].joints
        smplx_vertices = result_dict['smplx_output'].vertices
        with open(json_path_calib, 'r') as f:
            trans = torch.asarray(json.load(f)['trans'])

        global_smplx_joints = self.transform_3d_points_to_global_coord(smplx_joints, trans)
        global_smplx_vertices = self.transform_3d_points_to_global_coord(smplx_vertices, trans)
        if self.place_on_floor:
            floor_height_smplx_joints, _ = torch.min(global_smplx_joints.view(-1, 3), dim=0)
            global_smplx_joints[:, :, 1] -= floor_height_smplx_joints[1]
            floor_height_smplx_vertices, _ = torch.min(global_smplx_vertices.view(-1, 3), dim=0)
            global_smplx_vertices[:, :, 1] -= floor_height_smplx_vertices[1]
        result_dict['global_smplx_joints'] = global_smplx_joints
        result_dict['global_smplx_vertices'] = global_smplx_vertices
        result_dict['json_path_calib'] = json_path_calib
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
