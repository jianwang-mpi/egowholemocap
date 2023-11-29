# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os
import pickle

import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset

from mmpose.datasets.pipelines import Compose
from ...builder import DATASETS

from scipy.io import loadmat
from mmpose.core.evaluation.pose3d_eval import keypoint_mpjpe

@DATASETS.register_module()
class Mo2Cap2TestDataset(Dataset):
    weipeng_studio_img_path = r'/HPS/Mo2Cap2Plus1/static00/ExternalEgo/weipeng_studio/imgs'
    weipeng_studio_gt_path = r'/HPS/Mo2Cap2Plus/static00/Datasets/Mo2Cap2/data/test_data/weipeng_studio_gt.mat'
    olek_outdoor_img_path = r'/HPS/Mo2Cap2Plus1/static00/ExternalEgo/olek_outdoor/imgs'
    olek_outdoor_gt_path = r'/HPS/Mo2Cap2Plus/static00/Datasets/Mo2Cap2/data/test_data/olek_outdoor_gt.mat'

    def __init__(self, pipeline, data_cfg, test_mode=False):
        self.ann_info = data_cfg
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.data = self.load_dataset()

    def load_dataset(self):
        data = []
        weipeng_studio_gt_poses = loadmat(self.weipeng_studio_gt_path)['pose_gt']
        weipeng_studio_gt_poses = np.asarray(weipeng_studio_gt_poses) / 1000
        weipeng_start_frame = 386
        weipeng_end_frame = 3288
        weipeng_gt_poses = weipeng_studio_gt_poses[:weipeng_end_frame - weipeng_start_frame]
        weipeng_studio_image_names = natsorted(os.listdir(self.weipeng_studio_img_path))
        weipeng_studio_image_paths = [os.path.join(self.weipeng_studio_img_path, name) for name in
                                        weipeng_studio_image_names]
        assert len(weipeng_studio_image_paths) == len(weipeng_gt_poses)
        for i in range(len(weipeng_studio_image_paths)):
            data_item = {
                'image_file': weipeng_studio_image_paths[i],
                'keypoints_3d': weipeng_gt_poses[i],
                'keypoints_3d_visible': np.ones(weipeng_gt_poses[i].shape[0]),
                'seq_name': 'weipeng_studio',
                'frame_id': i,
                # 'data_name': 'mo2cap2'
            }
            data.append(data_item)

        olek_outdoor_gt_poses = loadmat(self.olek_outdoor_gt_path)['pose_gt']
        olek_outdoor_gt_poses = np.asarray(olek_outdoor_gt_poses) / 1000
        olek_start_frame = 157
        olek_end_frame = 2901
        olek_gt_poses = olek_outdoor_gt_poses[:olek_end_frame - olek_start_frame]
        olek_outdoor_image_names = natsorted(os.listdir(self.olek_outdoor_img_path))
        olek_outdoor_image_paths = [os.path.join(self.olek_outdoor_img_path, name) for name in
                                        olek_outdoor_image_names]
        assert len(olek_outdoor_image_paths) == len(olek_gt_poses)
        for i in range(len(olek_outdoor_image_paths)):
            data_item = {
                'image_file': olek_outdoor_image_paths[i],
                'keypoints_3d': olek_gt_poses[i],
                'keypoints_3d_visible': np.ones(olek_gt_poses[i].shape[0]),
                'seq_name': 'olek_outdoor',
                'frame_id': i,
                # 'data_name': 'mo2cap2'
            }
            data.append(data_item)
        return data

    def prepare_data(self, idx):
        data_item = self.data[idx]

        return data_item

    def evaluate(self, outputs, res_folder, metric=['pa-mpjpe', 'ba-mpjpe'], logger=None):
        # save to res_folder
        print(f'save to {res_folder}')
        with open(os.path.join(res_folder, 'results.pkl'), 'wb') as f:
            pickle.dump(outputs, f)
        eval_res_dict = {}
        pred_joints_3d_list = []
        gt_joints_3d_list = []
        gt_joints_visible_list = []
        for pred in outputs:
            pred_joints_3d_item = pred['keypoints_pred']

            if torch.is_tensor(pred_joints_3d_item):
                pred_joints_3d_item = pred_joints_3d_item.cpu().numpy()

            pred_joints_3d_list.extend(pred_joints_3d_item)
            img_meta_list = pred['img_metas']
            for img_meta_item in img_meta_list:
                gt_joints_3d_item = img_meta_item['keypoints_3d']
                gt_joints_3d_list.append(gt_joints_3d_item)
                gt_joints_visible_list.append(img_meta_item['keypoints_3d_visible'])

        pred_joints_3d = np.array(pred_joints_3d_list)
        gt_joints_3d = np.array(gt_joints_3d_list)
        gt_joints_visible = np.array(gt_joints_visible_list).astype(bool)
        assert len(pred_joints_3d[0]) == len(gt_joints_3d[0])

        pred_joints_3d_copy = copy.deepcopy(pred_joints_3d)
        gt_joints_3d_copy = copy.deepcopy(gt_joints_3d)
        eval_res_mpjpe = keypoint_mpjpe(pred_joints_3d_copy, gt_joints_3d_copy, gt_joints_visible, alignment='procrustes')
        eval_res_dict['pa-mpjpe'] = eval_res_mpjpe

        # now we estimate both!
        pred_joints_3d_copy = copy.deepcopy(pred_joints_3d)
        gt_joints_3d_copy = copy.deepcopy(gt_joints_3d)
        eval_res_pa_mpjpe = keypoint_mpjpe(
            pred_joints_3d_copy,
            gt_joints_3d_copy,
            gt_joints_visible,
            alignment='bone_length')
        eval_res_dict['ba-mpjpe'] = eval_res_pa_mpjpe

        return eval_res_dict

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        # print(f'get item {idx}')
        results = self.prepare_data(idx)
        results['ann_info'] = self.ann_info
        return self.pipeline(results)
