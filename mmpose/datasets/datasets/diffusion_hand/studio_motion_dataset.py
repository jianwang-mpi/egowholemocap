#  Copyright Jian Wang @ MPI-INF (c) 2023.
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
from mmpose.datasets.pipelines import Compose
from ...builder import DATASETS


@DATASETS.register_module()
class MocapStudioMotionDataset(Dataset):
    path_dict_new = {
        'new_jian1': {
            'path': r'/HPS/ScanNet/work/egocentric_view/25082022/jian1',
        },
        'new_jian2': {
            'path': r'/HPS/ScanNet/work/egocentric_view/25082022/jian2',
        },
        'new_diogo1': {
            'path': r'/HPS/ScanNet/work/egocentric_view/25082022/diogo1',
        },
        'new_diogo2': {
            'path': r'/HPS/ScanNet/work/egocentric_view/25082022/diogo2',
        },
    }
    path_dict_old = {'jian1': {
        'path': r'/HPS/ScanNet/work/egocentric_view/05082022/jian1',
    },
        'jian2': {
            'path': r'/HPS/ScanNet/work/egocentric_view/05082022/jian2',
        },
        'diogo1': {
            'path': r'/HPS/ScanNet/work/egocentric_view/05082022/diogo1',
        },
        'diogo2': {
            'path': r'/HPS/ScanNet/work/egocentric_view/05082022/diogo2',
        },
        'pranay2': {
            'path': r'/HPS/ScanNet/work/egocentric_view/05082022/pranay2',
        },

    }

    def __init__(self,
                 pipeline,
                 seq_len,
                 skip_frames,
                 split_sequence=True,
                 local=False,
                 test_mode=False,
                 sample_a_few=False,
                 data_cfg=None,
                 use_all_data=False,
                 ):
        self.seq_len = seq_len
        self.skip_frames = skip_frames
        self.pipeline = pipeline
        self.split_sequence = split_sequence
        self.data_cfg = copy.deepcopy(data_cfg)
        self.ann_info = {}

        self.load_config(self.data_cfg)
        self.path_dict = copy.deepcopy(self.path_dict_new)
        if use_all_data:
            self.path_dict.update(self.path_dict_old)


        self.local = local
        if local:
            for key in self.path_dict.keys():
                self.path_dict[key]['path'] = self.path_dict[key]['path'].replace('/HPS', 'X:')

        self.test_mode = test_mode

        self.studio_index, self.smplx_index = dset_to_body_model(dset='studio', model_type='smplx')

        self.data_info = self.load_annotations()
        self.pipeline = Compose(pipeline)

        if sample_a_few:
            print('!!!!!warning, only sample a few data for debug!!!!!!!')
            self.data_info = self.data_info[::100]

    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """
        self.ann_info = copy.deepcopy(data_cfg)

    def load_annotations(self):
        """Load data annotation."""
        print("start loading test file")
        data_info = []
        for seq_name in self.path_dict:
            base_path = self.path_dict[seq_name]['path']

            gt_path = os.path.join(base_path, 'local_pose_gt_with_hand.pkl')

            with open(gt_path, 'rb') as f:
                pose_gt_data = pickle.load(f)

            global_gt_pose_list = []
            local_gt_pose_list = []
            calib_board_pose_list = []

            for pose_gt_item in pose_gt_data:
                ext_id = pose_gt_item['ext_id']
                ego_keypoints_3d = pose_gt_item['ego_pose_gt']
                ext_keypoints_3d = pose_gt_item['ext_pose_gt']
                calib_board_pose = pose_gt_item['calib_board_pose']
                if ego_keypoints_3d is None:
                    print(f"None pose in base_path: {base_path} and ext_id: {ext_id}")
                    continue

                local_gt_pose_list.append(ego_keypoints_3d)
                global_gt_pose_list.append(ext_keypoints_3d)
                calib_board_pose_list.append(calib_board_pose)

            if self.split_sequence:
                assert len(local_gt_pose_list) == len(global_gt_pose_list) == len(calib_board_pose_list)
                local_gt_seq_list = self.split_into_sequences(local_gt_pose_list)
                global_gt_seq_list = self.split_into_sequences(global_gt_pose_list)
                calib_board_seq_list = self.split_into_sequences(calib_board_pose_list)
                assert len(local_gt_seq_list) == len(global_gt_seq_list) == len(calib_board_seq_list)
                for i in range(len(local_gt_seq_list)):
                    data_info.append(
                        {
                            'seq_name': seq_name,
                            'ego_keypoints_3d': local_gt_seq_list[i],
                            'ext_keypoints_3d': global_gt_seq_list[i],
                            'calib_board_pose': calib_board_seq_list[i],
                        }
                    )
            else:
                data_info.append(
                    {
                        'seq_name': seq_name,
                        'ego_keypoints_3d': local_gt_pose_list,
                        'ext_keypoints_3d': global_gt_pose_list,
                        'calib_board_pose': calib_board_pose_list,
                    }
                )
        return data_info

    def split_into_sequences(self, seq):
        seq_len = self.seq_len
        skip_frames = self.skip_frames
        result = []
        for i in range(0, len(seq) - seq_len + 1, skip_frames):
            result.append(seq[i:i + seq_len])
        return result

    def evaluate(self, outputs, res_folder, metric=['mpjpe', 'pa-mpjpe', 'pck'], logger=None):

        res_file = os.path.join(res_folder, 'results.pkl')
        with open(res_file, 'wb') as f:
            pickle.dump(outputs, f)
        evaluation_results = {'result': 0}

        return evaluation_results

    def prepare_data(self, idx):
        """Get data sample."""
        """Get data sample."""
        sequence = self.data_info[idx]

        # convert from renderpeople joints to smplx joints under the egocentric coordinate system

        studio_ego_joint_list = sequence['ego_keypoints_3d']
        studio_ego_joint_list = np.asarray(studio_ego_joint_list)
        ego_smplx_joint_list = np.zeros((studio_ego_joint_list.shape[0], 145, 3))
        ego_smplx_joint_list[:, self.smplx_index] = studio_ego_joint_list[:, self.studio_index]

        # convert from renderpeople joints to smplx joints under the world coordinate system
        studio_global_joint_list = sequence['ext_keypoints_3d']
        studio_global_joint_list = np.asarray(studio_global_joint_list)
        global_smplx_joint_list = np.zeros((studio_global_joint_list.shape[0], 145, 3))
        global_smplx_joint_list[:, self.smplx_index] = studio_global_joint_list[:, self.studio_index]

        result_dict = {
            'ego_smplx_joints': ego_smplx_joint_list,
            'global_smplx_joints': global_smplx_joint_list,
        }

        return result_dict

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.data_info)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        # print(f'get item {idx}')
        results = copy.deepcopy(self.prepare_data(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)
