# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from mmpose.core.evaluation.pose3d_eval import keypoint_mpjpe
from mmpose.datasets.pipelines import Compose
from mmpose.utils.visualization.skeleton import Skeleton
from .joint_converter import dset_to_body_model
from ...builder import DATASETS

from mmpose.core.evaluation.top_down_eval import pose_pck_accuracy, _get_max_preds, _get_softargmax_preds
@DATASETS.register_module()
class RenderpeopleMixamoHandTrainDataset(Dataset):
    allowed_metrics = ['pa-mpjpe', 'mpjpe', 'ba-mpjpe', 'pck', 'hand-mpjpe']

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False,
                 part_dataset=False
                 ):

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.data_cfg = copy.deepcopy(data_cfg)
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.part_dataset = part_dataset
        self.ann_info = {}

        self.load_config(self.data_cfg)
        self.skeleton = Skeleton(self.ann_info['camera_param_path'])

        self.data_info = self.load_annotations()
        self.pipeline = Compose(pipeline)



        if self.ann_info['joint_type'] == 'smplx':
            self.dst_idxs, self.model_idxs = dset_to_body_model(
                dset='renderpeople',
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

        data_list = []
        data_raw_list = data['data_list']
        for identity_name, identity_data in data_raw_list.items():
            print(f'load identity: {identity_name}')
            for seq_name, seq_data in tqdm(identity_data.items()):
                data_list.extend(seq_data)

        if self.part_dataset is True:
            # select part of the dataset for evaluation
            print('select part of the dataset for evaluation')
            data_list = data_list[:200]
        return data_list

    def evaluate(self, outputs, res_folder, metric=['mpjpe', 'pa-mpjpe'], logger=None):
        """Evaluate 3D keypoint results."""
        metrics = metric if isinstance(metric, list) else [metric]
        for metric in metrics:
            if metric not in self.allowed_metrics:
                raise KeyError(f'metric {metric} is not supported, supported metrics are {self.allowed_metrics}')

        # res_file = os.path.join(res_folder, 'results.pkl')
        # with open(res_file, 'wb') as f:
        #     pickle.dump(outputs, f)
        evaluation_results = {}
        for metric_name in metrics:
            evaluation_numbers = self._report_metric_and_save(outputs, metric_name, res_folder)
            evaluation_results = {**evaluation_results, **evaluation_numbers}

        return evaluation_results

    def prepare_data(self, idx):
        """Get data sample."""
        result = {}
        data = self.data_info[idx]
        result['image_file'] = os.path.join(self.img_prefix, data['img_path'])
        # get ext_id, seq_name
        image_path_list = os.path.normpath(data['img_path']).split(os.path.sep)
        image_id_str = os.path.splitext(image_path_list[-1])[0]
        result['ext_id'] = int(image_id_str)
        # get seq_name and identity name
        seq_name = image_path_list[-4] + '_' + image_path_list[-3]
        result['seq_name'] = seq_name
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
            result['left_hand_keypoints_3d'] = left_hand_keypoints_3d
            result['right_hand_keypoints_3d'] = right_hand_keypoints_3d
            result['left_hand_keypoints_3d_visible'] = left_hand_keypoints_3d_visible
            result['right_hand_keypoints_3d_visible'] = right_hand_keypoints_3d_visible

            result['ext_pose_gt'] = data['renderpeople_joints']
            result['ego_camera_pose'] = data['camera_info']

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

    def _report_metric_and_save(self, result_list, metric_name, res_folder):
        """Keypoint evaluation.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (MPJPE-PA)
        """
        eval_res_dict = {}
        if metric_name == 'mpjpe' or metric_name == 'pa-mpjpe' or metric_name == 'ba-mpjpe':
            pred_joints_3d_list = []
            gt_joints_3d_list = []
            gt_joints_visible_list = []
            for pred in result_list:
                pred_joints_3d_item = pred['keypoints_pred']

                if torch.is_tensor(pred_joints_3d_item):
                    pred_joints_3d_item = pred_joints_3d_item.cpu().numpy()

                # convert from model format to mo2cap2 format
                pred_joints_3d_item = self._convert_from_model_format_to_mo2cap2_format(pred_joints_3d_item,
                                                                  model_idxs=self.model_idxs,
                                                                  dst_idxs=self.dst_idxs)
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
            if metric_name == 'mpjpe':
                pred_joints_3d_copy = copy.deepcopy(pred_joints_3d)
                gt_joints_3d_copy = copy.deepcopy(gt_joints_3d)
                eval_res_mpjpe = keypoint_mpjpe(pred_joints_3d_copy, gt_joints_3d_copy, gt_joints_visible, alignment='none')
                eval_res_dict['mpjpe'] = eval_res_mpjpe

                # now we estimate both!
                pred_joints_3d_copy = copy.deepcopy(pred_joints_3d)
                gt_joints_3d_copy = copy.deepcopy(gt_joints_3d)
                eval_res_pa_mpjpe = keypoint_mpjpe(
                    pred_joints_3d_copy,
                    gt_joints_3d_copy,
                    gt_joints_visible,
                    alignment='procrustes')
                eval_res_dict['pa-mpjpe'] = eval_res_pa_mpjpe

            if metric_name == 'pa-mpjpe':
                pred_joints_3d_copy = copy.deepcopy(pred_joints_3d)
                gt_joints_3d_copy = copy.deepcopy(gt_joints_3d)
                eval_res = keypoint_mpjpe(
                    pred_joints_3d_copy,
                    gt_joints_3d_copy,
                    gt_joints_visible,
                    alignment='procrustes')
                eval_res_dict['pa-mpjpe'] = eval_res
            if metric_name == 'ba-mpjpe':
                pred_joints_3d_copy = copy.deepcopy(pred_joints_3d)
                gt_joints_3d_copy = copy.deepcopy(gt_joints_3d)
                eval_res = keypoint_mpjpe(
                    pred_joints_3d_copy,
                    gt_joints_3d_copy,
                    gt_joints_visible,
                    alignment='bone_length')
                eval_res_dict['ba-mpjpe'] = eval_res

            # save result to res folder
            if res_folder is not None:
                print(f'save to {res_folder}')
                with open(os.path.join(res_folder, 'results.pkl'), 'wb') as f:
                    pickle.dump(result_list, f)

        if metric_name == 'hand-mpjpe':
            # hand evaluation
            pred_left_hand_joint_list = []
            pred_right_hand_joint_list = []

            gt_left_hand_joint_list = []
            gt_right_hand_joint_list = []

            for pred in result_list:
                pred_left_hand_joint = pred['left_hands_preds']['joint_cam_transformed']
                pred_right_hand_joint = pred['right_hands_preds']['joint_cam_transformed']
                if torch.is_tensor(pred_left_hand_joint):
                    pred_left_hand_joint = pred_left_hand_joint.cpu().numpy()
                    pred_right_hand_joint = pred_right_hand_joint.cpu().numpy()
                pred_left_hand_joint_list.extend(pred_left_hand_joint)
                pred_right_hand_joint_list.extend(pred_right_hand_joint)

                img_meta_list = pred['img_metas']
                for img_meta_item in img_meta_list:
                    left_hand_keypoints_3d = img_meta_item['left_hand_keypoints_3d']
                    right_hand_keypoints_3d = img_meta_item['right_hand_keypoints_3d']
                    gt_left_hand_joint_list.append(left_hand_keypoints_3d)
                    gt_right_hand_joint_list.append(right_hand_keypoints_3d)
            pred_left_hand_joint_list = np.asarray(pred_left_hand_joint_list)
            pred_right_hand_joint_list = np.asarray(pred_right_hand_joint_list)
            gt_left_hand_joint_list = np.asarray(gt_left_hand_joint_list)
            gt_right_hand_joint_list = np.asarray(gt_right_hand_joint_list)

            mask = np.ones([pred_left_hand_joint_list.shape[0], 21], dtype=int).astype(bool)
            # calculate mpjpe
            left_hand_mpjpe = keypoint_mpjpe(pred_left_hand_joint_list, gt_left_hand_joint_list,
                                             mask, alignment='root')
            right_hand_mpjpe = keypoint_mpjpe(pred_right_hand_joint_list, gt_right_hand_joint_list,
                                             mask, alignment='root')
            left_hand_pa_mpjpe = keypoint_mpjpe(pred_left_hand_joint_list, gt_left_hand_joint_list,
                                             mask, alignment='procrustes')
            right_hand_pa_mpjpe = keypoint_mpjpe(pred_right_hand_joint_list, gt_right_hand_joint_list,
                                                mask, alignment='procrustes')

            # save the results
            if res_folder is not None:
                print(f'save to {res_folder}')
                with open(os.path.join(res_folder, 'results.pkl'), 'wb') as f:
                    pickle.dump(result_list, f)

            eval_res_dict = {
                'left_hand_mpjpe': left_hand_mpjpe,
                'right_hand_mpjpe': right_hand_mpjpe,
                'left_hand_pa_mpjpe': left_hand_pa_mpjpe,
                'right_hand_pa_mpjpe': right_hand_pa_mpjpe,
                'hand_mpjpe': (left_hand_mpjpe + right_hand_mpjpe) / 2,
                'hand_pa_mpjpe': (left_hand_pa_mpjpe + right_hand_pa_mpjpe) / 2,
            }

        if metric_name == 'pck':
            heatmaps_2d_pred = []
            heatmaps_2d_pred_original = []
            heatmaps_2d_gt = []
            keypoints_2d_gt_original = []
            image_file_list = []
            gt_joints_visible_list = []
            for pred in result_list:
                output_heatmap_item = pred['output_heatmap']
                heatmaps_2d_pred_original.extend(output_heatmap_item)
                # convert from model format to mo2cap2 format
                # output_heatmap_item = self._convert_from_model_format_to_mo2cap2_format(output_heatmap_item,
                #                                                                         model_idxs=self.model_idxs,
                #                                                                         dst_idxs=self.dst_idxs)
                heatmaps_2d_pred.extend(output_heatmap_item)
                img_meta_list = pred['img_metas']
                for img_meta_item in img_meta_list:
                    target = img_meta_item['target']
                    keypoints_2d = img_meta_item['keypoints_2d']
                    keypoints_2d_gt_original.append(keypoints_2d)
                    heatmaps_2d_gt.append(target)
                    gt_joints_visible_list.append(img_meta_item['keypoints_2d_visible'])
                    image_file_list.append(img_meta_item['image_file'])
            heatmaps_2d_pred = np.asarray(heatmaps_2d_pred)
            heatmaps_2d_pred_original = np.asarray(heatmaps_2d_pred_original)
            heatmaps_2d_gt = np.asarray(heatmaps_2d_gt)
            keypoints_2d_gt_original = np.asarray(keypoints_2d_gt_original)
            gt_joints_visible = np.asarray(gt_joints_visible_list).astype(np.bool)
            print(heatmaps_2d_pred.shape)
            print(heatmaps_2d_gt.shape)
            print(gt_joints_visible.shape)
            N, K, H, W = heatmaps_2d_pred.shape
            acc, avg_acc, cnt = pose_pck_accuracy(heatmaps_2d_pred, heatmaps_2d_gt, gt_joints_visible, thr=0.05)
            eval_res_dict['pck'] = avg_acc
            eval_res_dict['joint_pck'] = acc
            # save result to res folder
            if res_folder is not None:
                print(f'save to {res_folder}')
                # joints_2d_pred, _ = _get_max_preds(heatmaps_2d_pred_original)
                joints_2d_pred, _ = _get_softargmax_preds(heatmaps_2d_pred_original)
                with open(os.path.join(res_folder, 'results.pkl'), 'wb') as f:
                    pickle.dump({'joints_2d_pred': joints_2d_pred, 'image_file': image_file_list,
                                 'joints_2d_gt': keypoints_2d_gt_original}, f)

        if len(eval_res_dict) == 0:
            raise KeyError(f'metric {metric_name} is not supported, supported metrics are {self.allowed_metrics}')
        return eval_res_dict

