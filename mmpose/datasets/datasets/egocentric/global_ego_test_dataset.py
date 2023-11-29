# Copyright (c) OpenMMLab. All rights reserved.
import copy
import pickle
import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset
import os

from tqdm import tqdm

from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose
from mmpose.utils.visualization.skeleton import Skeleton
from mmpose.core.evaluation.pose3d_eval import keypoint_mpjpe
from mmpose.core.evaluation.top_down_eval import pose_pck_accuracy, _get_max_preds
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
import json
from ...builder import DATASETS

@DATASETS.register_module()
class GlobalEgoTestDataset(Dataset):

    allowed_metrics = ['pa-mpjpe', 'ba-mpjpe', 'pck']

    path_dict = {
        'jian3': {
            'gt_path': r'/HPS/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/jian3/jian3.pkl',
            'start_frame': 557,
            'end_frame': 1857,
            "predicted_path": r'/HPS/Mo2Cap2Plus1/static00/EgocentricData/REC08102020/jian3'
        },
        'studio-jian1': {
            'gt_path': r'/HPS/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/studio-jian1/jian1.pkl',
            'start_frame': 503,
            'end_frame': 3603,
            "predicted_path": r'/HPS/Mo2Cap2Plus1/static00/EgocentricData/REC23102020/studio-jian1'
        },
        'studio-jian2': {
            'gt_path': r'/HPS/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/studio-jian2/jian2.pkl',
            'start_frame': 600,
            'end_frame': 3400,
            "predicted_path": r'/HPS/Mo2Cap2Plus1/static00/EgocentricData/REC23102020/studio-jian2'
        },
        'studio-lingjie1': {
            'gt_path': r'/HPS/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/studio-lingjie1/lingjie1.pkl',
            'start_frame': 551,
            'end_frame': 3251,
            "predicted_path": r'/HPS/Mo2Cap2Plus1/static00/EgocentricData/REC23102020/studio-lingjie1'
        },
        'studio-lingjie2': {
            'gt_path': r'/HPS/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/studio-lingjie2/lingjie2.pkl',
            'start_frame': 438,
            'end_frame': 2738,
            "predicted_path": r'/HPS/Mo2Cap2Plus1/static00/EgocentricData/REC23102020/studio-lingjie2'
        }
    }

    def __init__(self,
                 data_cfg,
                 pipeline,
                 local=False,
                 test_mode=False):

        self.data_cfg = copy.deepcopy(data_cfg)
        self.local = local
        if local:
            for key in self.path_dict.keys():
                self.path_dict[key]['path'] = self.path_dict[key]['path'].replace('/HPS', 'X:')
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.ann_info = {}

        self.load_config(self.data_cfg)
        self.skeleton = Skeleton(self.ann_info['camera_param_path'])

        self.dst_idxs, self.model_idxs = dset_to_body_model(dset='mo2cap2', model_type=self.ann_info['joint_type'],
                use_face_contour=False)

        self.data_info = self.load_annotations()
        self.pipeline = Compose(pipeline)


    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """
        self.ann_info = copy.deepcopy(data_cfg)
        self.ann_info['image_size'] = np.asarray(self.ann_info['image_size'])


    def load_gt_data(self, gt_path, start_frame, end_frame, mat_start_frame):
        with open(gt_path, 'rb') as f:
            pose_gt = pickle.load(f)
        clip = []
        for i in range(start_frame, end_frame):
            clip.append(pose_gt[i - mat_start_frame])

        skeleton_list = clip

        return np.asarray(skeleton_list)


    def load_annotations(self):
        """Load data annotation."""
        print("start loading test file")
        data_info = []
        for seq_name in self.path_dict:
            base_path = self.path_dict[seq_name]['predicted_path']
            print('loading {}'.format(base_path))
            image_dir = os.path.join(base_path, 'imgs')
            gt_path = self.path_dict[seq_name]['gt_path']
            start_frame = self.path_dict[seq_name]['start_frame']
            end_frame = self.path_dict[seq_name]['end_frame']

            gt_pose_list = self.load_gt_data(gt_path, start_frame, end_frame, start_frame)
            image_name_list = natsorted(os.listdir(image_dir))
            image_name_list = image_name_list[start_frame: end_frame]
            for image_name, gt_pose in tqdm(zip(image_name_list, gt_pose_list)):
                image_path = os.path.join(image_dir, image_name)
                data_info.append(
                    dict(
                        image_file=image_path,
                        keypoints_3d=gt_pose,
                        keypoints_3d_visible=np.ones([15], dtype=np.float32),
                        seq_name=seq_name,
                    )
                )

        return data_info

    def evaluate(self, outputs, res_folder, metric=['mpjpe', 'pa-mpjpe', 'pck'], logger=None):
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


    def _report_metric_and_save(self, result_list, metric_name, res_folder):
        """Keypoint evaluation.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (MPJPE-PA)
        """
        eval_res_dict = {}
        if 'mpjpe' == metric_name or 'pa-mpjpe' == metric_name or 'ba-mpjpe' == metric_name:
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
            if 'mpjpe' == metric_name:
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

            if 'pa-mpjpe' == metric_name:
                pred_joints_3d_copy = copy.deepcopy(pred_joints_3d)
                gt_joints_3d_copy = copy.deepcopy(gt_joints_3d)
                eval_res = keypoint_mpjpe(
                    pred_joints_3d_copy,
                    gt_joints_3d_copy,
                    gt_joints_visible,
                    alignment='procrustes')
                eval_res_dict['pa-mpjpe'] = eval_res

                # now we estimate both!
                pred_joints_3d_copy = copy.deepcopy(pred_joints_3d)
                gt_joints_3d_copy = copy.deepcopy(gt_joints_3d)
                eval_res_pa_mpjpe = keypoint_mpjpe(
                    pred_joints_3d_copy,
                    gt_joints_3d_copy,
                    gt_joints_visible,
                    alignment='bone_length')
                eval_res_dict['ba-mpjpe'] = eval_res_pa_mpjpe
            if 'ba-mpjpe' == metric_name:
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

        if 'pck' in metric_name:
            heatmaps_2d_pred = []
            heatmaps_2d_pred_original = []
            heatmaps_2d_gt = []
            image_file_list = []
            gt_joints_visible_list = []
            for pred in result_list:
                output_heatmap_item = pred['output_heatmap']
                heatmaps_2d_pred_original.extend(output_heatmap_item)
                # convert from model format to mo2cap2 format
                output_heatmap_item = self._convert_from_model_format_to_mo2cap2_format(output_heatmap_item,
                                                                                        model_idxs=self.model_idxs,
                                                                                        dst_idxs=self.dst_idxs)
                heatmaps_2d_pred.extend(output_heatmap_item)
                img_meta_list = pred['img_metas']
                for img_meta_item in img_meta_list:
                    target = img_meta_item['target']
                    heatmaps_2d_gt.append(target)
                    gt_joints_visible_list.append(img_meta_item['keypoints_2d_visible'])
                    image_file_list.append(img_meta_item['image_file'])
            heatmaps_2d_pred = np.asarray(heatmaps_2d_pred)
            heatmaps_2d_pred_original = np.asarray(heatmaps_2d_pred_original)
            heatmaps_2d_gt = np.asarray(heatmaps_2d_gt)
            gt_joints_visible = np.asarray(gt_joints_visible_list).astype(np.bool)
            print(heatmaps_2d_pred.shape)
            print(heatmaps_2d_gt.shape)
            N, K, H, W = heatmaps_2d_pred.shape
            mask = np.ones((N, K)).astype(np.bool)
            acc, avg_acc, cnt = pose_pck_accuracy(heatmaps_2d_pred, heatmaps_2d_gt, gt_joints_visible, thr=0.05)
            eval_res_dict['pck'] = avg_acc
            eval_res_dict['joint_pck'] = acc
            # save result to res folder
            if res_folder is not None:
                print(f'save to {res_folder}')
                joints_2d_pred, _ = _get_max_preds(heatmaps_2d_pred_original)
                with open(os.path.join(res_folder, 'results.pkl'), 'wb') as f:
                    pickle.dump({'joints_2d_pred': joints_2d_pred, 'image_file': image_file_list}, f)

        if len(eval_res_dict) == 0:
            raise KeyError(f'metric {metric_name} is not supported, supported metrics are {self.allowed_metrics}')
        return eval_res_dict

    def _convert_from_model_format_to_mo2cap2_format(self, pred_joints, model_idxs, dst_idxs):
        mo2cap2_shape = list(pred_joints.shape)
        mo2cap2_shape[1] = 15
        mo2cap2_joint_batch = np.empty(mo2cap2_shape)

        mo2cap2_joint_batch[:, dst_idxs] = pred_joints[:, model_idxs]
        return mo2cap2_joint_batch



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
