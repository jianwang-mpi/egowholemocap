# Copyright (c) OpenMMLab. All rights reserved.
import copy
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose
from mmpose.utils.visualization.skeleton import Skeleton
from mmpose.core.evaluation.pose3d_eval import keypoint_mpjpe
from mmpose.core.evaluation.top_down_eval import pose_pck_accuracy, _get_max_preds
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
import json
from natsort import natsorted
from ...builder import DATASETS

@DATASETS.register_module()
class MocapStudioFinetuneDataset(Dataset):

    allowed_metrics = ['pa-mpjpe', 'mpjpe', 'ba-mpjpe', 'pck']

    path_dict = {
        'jian1': {
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

    def get_gt_data(self, seq_name):
        print("start loading test file")
        base_path = self.path_dict[seq_name]['path']

        img_data_path = os.path.join(base_path, 'imgs')
        gt_path = os.path.join(base_path, 'local_pose_gt.pkl')
        depth_path = os.path.join(base_path, 'rendered', 'depths')
        syn_path = os.path.join(base_path, 'syn.json')

        with open(syn_path, 'r') as f:
            syn_data = json.load(f)

        ego_start_frame = syn_data['ego']
        ext_start_frame = syn_data['ext']

        image_path_list = []
        img_names = os.listdir(img_data_path)
        # print(self.data_path)
        img_names = natsorted(img_names)
        for img_name in img_names:
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                img_path = os.path.join(img_data_path, img_name)
                # img = cv2.imread(os.path.join(self.data_path, img_name))
                image_path_list.append(img_path)
        image_path_list = image_path_list[ego_start_frame:]

        def load_gt_data(gt_path):
            with open(gt_path, 'rb') as f:
                pose_gt_data = pickle.load(f)
            pose_gt_list = []
            for pose_item in pose_gt_data:
                pose_gt_list.append(pose_item['ego_pose_gt'])

            return np.asarray(pose_gt_list)

        def load_depth_data(depth_path):
            depth_name_list = natsorted(os.listdir(depth_path))
            depth_path_li = []
            for depth_name in depth_name_list:
                depth_full_path = os.path.join(depth_path, depth_name, 'Image0001.exr')
                depth_path_li.append(depth_full_path)
            # depth_path_li = depth_path_li[ext_start_frame:]
            return depth_path_li

        gt_pose_list = load_gt_data(gt_path)
        depth_path_list = load_depth_data(depth_path)

        if len(image_path_list) != len(gt_pose_list) or len(gt_pose_list) != len(depth_path_list):
            print('length of egocentric image: {}'.format(len(image_path_list)))
            print('length of gt pose: {}'.format(len(gt_pose_list)))
            print('length of depth map: {}'.format(len(depth_path_list)))
            min_len = min(len(image_path_list), min(len(gt_pose_list), len(depth_path_list)))
            image_path_list = image_path_list[:min_len]
            gt_pose_list = gt_pose_list[:min_len]
            depth_path_list = depth_path_list[:min_len]

        return image_path_list, gt_pose_list, depth_path_list

    def load_annotations(self):
        """Load data annotation."""
        print("start loading test file")
        data_info = []
        for seq_name in self.path_dict.keys():
            image_data, gt_pose_list, depth_path_list = self.get_gt_data(seq_name)
            for image_path, gt_pose in zip(image_data, gt_pose_list):
                keypoints_3d_visible = np.ones([15], dtype=np.float32)
                data_info.append(
                    {
                        'image_file': image_path,
                        'keypoints_3d': gt_pose,
                        'keypoints_3d_visible': keypoints_3d_visible
                    }
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
        if metric_name in ['mpjpe', 'pa-mpjpe', 'ba-mpjpe']:
            pred_joints_3d_list = []
            gt_joints_3d_list = []
            gt_joints_visible_list = []
            for pred in result_list:
                pred_joints_3d_item = pred['keypoints_pred']

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
            # N, K, C = pred_joints_3d.shape
            # gt_joints_visible = np.ones((N, K)).astype(bool)
            if metric_name == 'mpjpe':
                eval_res = keypoint_mpjpe(pred_joints_3d, gt_joints_3d, gt_joints_visible, alignment='none')
            elif metric_name == 'pa-mpjpe':
                eval_res = keypoint_mpjpe(
                    pred_joints_3d,
                    gt_joints_3d,
                    gt_joints_visible,
                    alignment='procrustes')
            elif metric_name == 'ba-mpjpe':
                eval_res = keypoint_mpjpe(
                    pred_joints_3d,
                    gt_joints_3d,
                    gt_joints_visible,
                    alignment='bone_length')
            else:
                raise KeyError(f'metric {metric_name} is not supported, supported metrics are {self.allowed_metrics}')
            eval_res = {metric_name: eval_res}

            # save result to res folder
            if res_folder is not None:
                print(f'save to {res_folder}')
                with open(os.path.join(res_folder, 'results.pkl'), 'wb') as f:
                    pickle.dump(result_list, f)

        elif metric_name in ['pck']:
            heatmaps_2d_pred = []
            heatmaps_2d_pred_original = []
            heatmaps_2d_gt = []
            image_file_list = []
            gt_joints_visible_list = []
            for pred in result_list:
                output_heatmap_item = pred['output_heatmap']
                heatmaps_2d_pred_original.extend(output_heatmap_item)
                # convert from model format to mo2cap2 format
                output_heatmap_item = self._convert_from_model_format_to_mo2cap2_format(output_heatmap_item.detach().cpu().numpy(),
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
            eval_res = {
                'pck': avg_acc,
                'joint_pck': acc
            }
            # save result to res folder
            if res_folder is not None:
                print(f'save to {res_folder}')
                joints_2d_pred, _ = _get_max_preds(heatmaps_2d_pred_original)
                with open(os.path.join(res_folder, 'results.pkl'), 'wb') as f:
                    pickle.dump({'joints_2d_pred': joints_2d_pred, 'image_file': image_file_list}, f)
        else:
            raise KeyError(f'metric {metric_name} is not supported, supported metrics are {self.allowed_metrics}')
        return eval_res

    def _convert_from_model_format_to_mo2cap2_format(self, pred_joints, model_idxs, dst_idxs):
        mo2cap2_shape = list(pred_joints.shape)
        mo2cap2_shape[1] = 15
        mo2cap2_joint_batch = np.empty(mo2cap2_shape)

        mo2cap2_joint_batch[:, dst_idxs] = pred_joints[:, model_idxs]
        return mo2cap2_joint_batch

    def prepare_data(self, idx):
        """Get data sample."""
        result = {}
        data = self.data_info[idx]
        result['image_file'] = data['image_file']
        # print(data['image_file'])
        if self.ann_info['joint_type'] == 'mo2cap2':
            result['keypoints_3d'] = data['keypoints_3d']
            N_joints, _ = data['keypoints_3d'].shape
            result['keypoints_3d_visible'] = np.ones([N_joints], dtype=np.float32)
        elif self.ann_info['joint_type'] == 'smplx':
            keypoints3d = np.zeros([127, 3], dtype=np.float32)
            keypoints3d_visible = np.zeros([127], dtype=np.float32)
            keypoints = data['keypoints_3d']
            # convert from renderpeople to smplx joint
            keypoints3d[self.model_idxs] = keypoints[self.dst_idxs]
            keypoints3d_visible[self.model_idxs] = 1
            result['keypoints_3d'] = keypoints3d
            result['keypoints_3d_visible'] = keypoints3d_visible

        result['pose'] = np.zeros((21 * 3), dtype=np.float32)
        result['beta'] = np.zeros((10,), dtype=np.float32)

        result['has_smpl'] = 0
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
