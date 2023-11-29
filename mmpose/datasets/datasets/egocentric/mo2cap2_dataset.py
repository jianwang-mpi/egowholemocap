# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from mmpose.datasets.pipelines import Compose
from ...builder import DATASETS


@DATASETS.register_module()
class Mo2Cap2Dataset(Dataset):
    '''
    The heatmap sequence:
    [Neck, RightShoulder, RightElbow, RightWrist, LeftShoulder, LeftElbow, LeftWrist,
    RightHip, RightKnee, RightAnkle, RightFoot, LeftHip, LeftKnee, LeftAnkle, LeftFoot]
    '''

    def __init__(self, data_path, data_cfg, pipeline, test_mode=False):
        self.data_path = data_path
        self.data_cfg = copy.deepcopy(data_cfg)
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        self.load_config(self.data_cfg)

        with open(os.path.join(self.data_path, 'filenames.json')) as f:
            self.training_files = json.load(f)

    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """
        self.ann_info = data_cfg
        self.ann_info['image_size'] = np.asarray(self.ann_info['image_size'])

    def prepare_data(self, idx):
        training_file_name = self.training_files[idx]
        training_file_path = os.path.join(self.data_path, training_file_name)
        with open(training_file_path, 'rb') as f:
            d = pickle.load(f)

        # use full body data
        img = d['Image'] / 255.
        assert img.shape[-1] == 320
        # cut out the center of the image
        img = img[:, :, 32: -32]
        # bgr to rgb
        img = img[::-1, :, :]
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = torch.from_numpy(img).float()

        heatmap = d['Heatmap'] / 255.
        annot3d = d['Annot3D'] / 1000.

        heatmap = torch.from_numpy(heatmap).float()

        keypoints_3d_visible = np.ones((annot3d.shape[0], ))

        result = {
            'img': img,
            'heatmap': heatmap,
            'keypoints_3d': annot3d,
            'keypoints_3d_visible': keypoints_3d_visible,
        }

        return result

    def evaluate(self, outputs, res_folder, metric=['mpjpe', 'pa-mpjpe'], logger=None):
        # dummy function
        return {'result': 0}

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.training_files)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        # print(f'get item {idx}')
        results = self.prepare_data(idx)
        results['ann_info'] = self.ann_info
        return self.pipeline(results)
