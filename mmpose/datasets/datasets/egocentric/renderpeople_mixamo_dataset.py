# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import pickle

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from mmpose.core.evaluation.pose3d_eval import keypoint_mpjpe
from mmpose.datasets.pipelines import Compose
from mmpose.utils.visualization.skeleton import Skeleton
from .joint_converter import dset_to_body_model
from ...builder import DATASETS


@DATASETS.register_module()
class RenderpeopleMixamoDataset(Dataset):
    allowed_metrics = ['pa-mpjpe', 'mpjpe', 'ba-mpjpe']

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dset='renderpeople_old',
                 test_mode=False):

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.data_cfg = copy.deepcopy(data_cfg)
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.dset = dset
        self.ann_info = {}

        self.load_config(self.data_cfg)
        self.skeleton = Skeleton(self.ann_info['camera_param_path'])

        self.data_info = self.load_annotations()
        self.pipeline = Compose(pipeline)



        if self.ann_info['joint_type'] == 'smplx':
            self.dst_idxs, self.model_idxs = dset_to_body_model(
                dset=dset,
                model_type='smplx',
                use_face_contour=False)

    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """
        self.ann_info = copy.deepcopy(data_cfg)
        self.ann_info['image_size'] = np.asarray(self.ann_info['image_size'])

    def load_annotations(self):
        """Load data annotation."""
        with open(self.ann_file, 'rb') as f:
            data = pickle.load(f)

        if self.dset == 'renderpeople_old':
            # old renderpeople dataset format
            data_list = data['data_list']
        elif self.dset == 'renderpeople':
            # new renderpeople data format
            data_list = []
            data_raw_list = data['data_list']
            for identity_name, identity_data in data_raw_list.items():
                print(f'load identity: {identity_name}')
                for seq_name, seq_data in tqdm(identity_data.items()):
                    data_list.extend(seq_data)
        else:
            raise Exception('dset type is incorrect')

        if self.test_mode is True:
            # select part of the dataset for evaluation
            data_list = data_list[:200]
        return data_list

    def evaluate(self, outputs, res_folder, metric=['mpjpe', 'pa-mpjpe'], logger=None):
        res_file = os.path.join(res_folder, 'results.pkl')
        with open(res_file, 'wb') as f:
            pickle.dump(outputs, f)

        return {'result': 0}

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'wb') as f:
            pickle.dump(keypoints, f)

    def _report_metric(self, res_file, metric_name):
        """Keypoint evaluation.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (MPJPE-PA)
        """
        with open(res_file, 'rb') as fin:
            preds = pickle.load(fin)
        assert len(preds) == len(self.data_info)

        pred_joints_3d = [pred['keypoints'] for pred in preds]
        gt_joints_3d = [item['joints_3d'] for item in self.data_info]

        pred_joints_3d = np.array(pred_joints_3d)
        gt_joints_3d = np.array(gt_joints_3d)

        assert len(pred_joints_3d[0]) == len(gt_joints_3d[0])

        # we only evaluate on 14 lsp joints
        if metric_name == 'mpjpe':
            eval_res = keypoint_mpjpe(pred_joints_3d, gt_joints_3d, alignment='none')
        elif metric_name == 'pa-mpjpe':
            eval_res = keypoint_mpjpe(
                pred_joints_3d,
                gt_joints_3d,
                alignment='procrustes')
        elif metric_name == 'ba-mpjpe':
            eval_res = keypoint_mpjpe(
                pred_joints_3d,
                gt_joints_3d,
                alignment='bone_length')
        else:
            raise KeyError(f'metric {metric_name} is not supported, supported metrics are {self.allowed_metrics}')
        eval_res = {metric_name: eval_res}
        return eval_res

    def prepare_data(self, idx):
        """Get data sample."""
        result = {}
        data = self.data_info[idx]
        result['image_file'] = os.path.join(self.img_prefix, data['img_path'])
        result['depth_file'] = os.path.join(self.img_prefix, data['depth_path'])
        result['seg_file'] = os.path.join(self.img_prefix, data['seg_path'])
        if self.ann_info['joint_type'] == 'mo2cap2':
            result['keypoints_3d'] = data['mo2cap2_local_joints']
            N_joints, _ = data['mo2cap2_local_joints'].shape
            result['keypoints_3d_visible'] = np.ones([N_joints], dtype=np.float32)
        elif self.ann_info['joint_type'] == 'renderpeople':
            keypoints_3d = data['renderpeople_local_joints']
            N_joints, _ = keypoints_3d.shape
            keypoints_3d_visible = np.ones([N_joints], dtype=np.float32)

            left_hand_keypoints_3d = np.concatenate([keypoints_3d[22:23], keypoints_3d[33:33 + 20]], axis=0)
            right_hand_keypoints_3d = np.concatenate([keypoints_3d[23:24], keypoints_3d[53:53 + 20]], axis=0)
            left_hand_keypoints_3d_visible = np.concatenate([keypoints_3d_visible[22:23],
                                                             keypoints_3d_visible[33:33 + 20]], axis=0)
            right_hand_keypoints_3d_visible = np.concatenate([keypoints_3d_visible[23:24],
                                                              keypoints_3d_visible[53:53 + 20]], axis=0)

            # if set only hand = True, then only use hand keypoints
            if 'only_hand' in self.ann_info.keys() and self.ann_info['only_hand'] is True:
                keypoints_3d = np.concatenate([left_hand_keypoints_3d, right_hand_keypoints_3d], axis=0)
                keypoints_3d_visible = np.concatenate([left_hand_keypoints_3d_visible,
                                                       right_hand_keypoints_3d_visible], axis=0)

            result['keypoints_3d'] = keypoints_3d
            result['keypoints_3d_visible'] = keypoints_3d_visible

        elif self.ann_info['joint_type'] == 'smplx':
            keypoints3d = np.zeros([127, 3], dtype=np.float32)
            keypoints3d_visible = np.zeros([127], dtype=np.float32)
            keypoints = data['renderpeople_local_joints']
            # convert from renderpeople to smplx joint
            keypoints3d[self.model_idxs] = keypoints[self.dst_idxs]
            keypoints3d_visible[self.model_idxs] = 1
            result['keypoints_3d'] = keypoints3d
            result['keypoints_3d_visible'] = keypoints3d_visible

        # return bbox of hands


        result['body_pose'] = np.zeros((21 * 3), dtype=np.float32)
        result['betas'] = np.zeros((10,), dtype=np.float32)

        result['has_smpl'] = 0
        return result

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.data_info)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        results = copy.deepcopy(self.prepare_data(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)
