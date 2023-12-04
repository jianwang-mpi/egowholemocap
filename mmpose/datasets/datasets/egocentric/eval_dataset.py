# Copyright (c) OpenMMLab. All rights reserved.
import copy
import pickle
import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset
import os
from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose
from mmpose.utils.visualization.skeleton import Skeleton
from mmpose.core.evaluation.pose3d_eval import keypoint_mpjpe
from mmpose.core.evaluation.top_down_eval import pose_pck_accuracy, _get_max_preds
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
import json
from ...builder import DATASETS

@DATASETS.register_module()
class EvalDataset(Dataset):

    def __init__(self,
                 data_cfg,
                 pipeline,
                 local=False,
                 test_mode=False):

        self.data_cfg = copy.deepcopy(data_cfg)
        self.local = local
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.ann_info = {}

        self.load_config(self.data_cfg)

        self.data_info = self.load_annotations()
        self.pipeline = Compose(pipeline)


    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """
        self.ann_info = copy.deepcopy(data_cfg)
        self.ann_info['image_size'] = np.asarray(self.ann_info['image_size'])


    def load_annotations(self):
        """Load data annotation."""
        print("start loading test file")
        data_info = []
        image_dir = self.ann_info['img_dir']
        image_list = natsorted(os.listdir(image_dir))
        image_list = [image_name for image_name in image_list if '.png' in image_name or '.jpg' in image_name]

        for image_name in image_list:
            data_info.append(
                {
                    'image_file': os.path.join(image_dir, image_name),
                }
            )
        return data_info

    def evaluate(self, outputs, res_folder, save_name=None, metric=None, logger=None):
        # save the results
        if res_folder is not None:
            print(f'\nsave to {res_folder}\n')
            with open(os.path.join(res_folder, f'outputs.pkl'), 'wb') as f:
                pickle.dump(outputs, f)
        return {'mpjpe': 0}

    def prepare_data(self, idx):
        """Get data sample."""
        result = self.data_info[idx]
        return result

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.data_info)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        # print(f'get item {idx}')
        results = copy.deepcopy(self.prepare_data(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)
