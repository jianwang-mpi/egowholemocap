# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from mmpose.core.evaluation.pose3d_eval import keypoint_mpjpe
from mmpose.datasets.pipelines import Compose
from mmpose.utils.visualization.skeleton import Skeleton
from .joint_converter import dset_to_body_model
from ...builder import DATASETS


@DATASETS.register_module()
class SyntheticSMPLDataset(Dataset):
    allowed_metrics = ['pa-mpjpe', 'mpjpe', 'ba-mpjpe']

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.data_cfg = copy.deepcopy(data_cfg)
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.ann_info = {}

        self.load_config(self.data_cfg)
        self.skeleton = Skeleton(self.ann_info['camera_param_path'])

        self.data_info = self.load_annotations()
        self.pipeline = Compose(pipeline)

        # if source body model type is the same as the return model type, use the beta parameter,
        # if not, do not use the beta parameter
        # if self.ann_info['joint_type'] == 'smpl':
        #     self.use_betas = True
        # else:
        #     self.use_betas = False
        self.dst_idxs, self.model_idxs = dset_to_body_model(
            dset='smpl',
            model_type=self.ann_info['joint_type'],
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
        data_list = data['data_list']
        if self.test_mode is True:
            # select part of the dataset for evaluation
            data_list = data_list[:200]
        return data_list

    def evaluate(self, outputs, res_folder, metric=['mpjpe', 'pa-mpjpe'], logger=None):
        res_file = os.path.join(res_folder, 'results.pkl')
        with open(res_file, 'wb') as f:
            pickle.dump(outputs, f)

        return {'result': 0}

    @DeprecationWarning
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
        # result['seg_file'] = os.path.join(self.img_prefix, data['seg_path'])


        if self.ann_info['joint_type'] == 'mo2cap2':
            keypoints3d = np.zeros([15, 3], dtype=np.float32)
            keypoints3d_visible = np.zeros([15], dtype=np.float32)
        elif self.ann_info['joint_type'] == 'renderpeople':
            keypoints3d = np.zeros([55, 3], dtype=np.float32)
            keypoints3d_visible = np.zeros([55], dtype=np.float32)
        elif self.ann_info['joint_type'] == 'smplx':
            keypoints3d = np.zeros([127, 3], dtype=np.float32)
            keypoints3d_visible = np.zeros([127], dtype=np.float32)
        else:
            raise Exception("joint type does not support")

        keypoints = data['smpl_local_joints']
        smpl_parameters = data['smpl_params']
        # convert from renderpeople to smplx joint
        keypoints3d[self.model_idxs] = keypoints[self.dst_idxs]
        keypoints3d_visible[self.model_idxs] = 1
        result['keypoints_3d'] = keypoints3d
        result['keypoints_3d_visible'] = keypoints3d_visible

        # convert smpl parameter to smplx parameter
        smpl_pose = smpl_parameters['pose']
        smpl_betas = smpl_parameters['shape']
        smpl_transl = smpl_parameters['trans']


        smplx_body_pose = smpl_pose[3: 66].astype(np.float32)
        smplx_global_orient = smpl_pose[0: 3].astype(np.float32)
        smplx_betas = smpl_betas.astype(np.float32)

        result['body_pose'] = smplx_body_pose
        result['betas'] = smplx_betas
        result['global_orient_world'] = smplx_global_orient
        result['transl_world'] = smpl_transl.astype(np.float32)

        result['has_smpl'] = 1
        return result

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.data_info)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        results = copy.deepcopy(self.prepare_data(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)
