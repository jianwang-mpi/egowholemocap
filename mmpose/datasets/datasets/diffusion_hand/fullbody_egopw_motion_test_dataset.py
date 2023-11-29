#  Copyright Jian Wang @ MPI-INF (c) 2023.
import copy
import os
import pickle

import numpy as np
import open3d
import torch
from natsort import natsorted
from torch.utils.data import Dataset

from mmpose.data.keypoints_mapping.hml import hml_joint_names
from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_joint_names
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.diffusion.mo2cap2_keypoints_to_hml3d import KeypointsToHumanML3D
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
from mmpose.datasets.pipelines import Compose


@DATASETS.register_module()
class FullBodyEgoPwMotionTestDataset(Dataset):
    frame_rate = 25

    def __init__(self, egopw_pred_data_pkl_path,
                 seq_len,
                 skip_frames,
                 pipeline,
                 output_frame_rate=25,
                 split_sequence=True,
                 test_mode=False):
        self.egopw_pred_data_pkl_path = egopw_pred_data_pkl_path
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.split_sequence = split_sequence
        self.skip_frames = skip_frames

        self.output_frame_rate = output_frame_rate
        self.seq_len = seq_len

        self.mano_left_idxs, self.smplx_idxs_mano_left = dset_to_body_model(dset='mano_left', model_type='smplx')
        self.mano_right_idxs, self.smplx_idxs_mano_right = dset_to_body_model(dset='mano_right', model_type='smplx')
        self.mo2cap2_idxs, self.smplx_idxs_mo2cap2 = dset_to_body_model(dset='mo2cap2', model_type='smplx')

        self.data = self.load_annotation()


    def __len__(self):
        return len(self.data)

    def load_network_pred_pkl(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        pred_joints_3d_list = []
        pred_left_hand_joint_3d_list = []
        pred_right_hand_joint_3d_list = []
        gt_joints_3d_list = []
        image_file_path_list = []
        for pred in data:
            pred_joints_3d_item = pred['body_pose_results']['keypoints_pred']

            pred_left_hand_joint_item = pred['left_hands_preds']['joint_cam_transformed']
            pred_right_hand_joint_item = pred['right_hands_preds']['joint_cam_transformed']

            if torch.is_tensor(pred_joints_3d_item):
                pred_joints_3d_item = pred_joints_3d_item.cpu().numpy()
            if torch.is_tensor(pred_left_hand_joint_item):
                pred_left_hand_joint_item = pred_left_hand_joint_item.cpu().numpy()
                pred_right_hand_joint_item = pred_right_hand_joint_item.cpu().numpy()

            pred_joints_3d_list.extend(pred_joints_3d_item)
            pred_left_hand_joint_3d_list.extend(pred_left_hand_joint_item)
            pred_right_hand_joint_3d_list.extend(pred_right_hand_joint_item)
            img_meta_list = pred['img_metas']
            for img_meta_item in img_meta_list:
                gt_joints_3d_item = img_meta_item['keypoints_3d']
                image_file_path = img_meta_item['image_file']
                gt_joints_3d_list.append(gt_joints_3d_item)
                image_file_path_list.append(image_file_path)

        gt_joints_3d_list = np.array(gt_joints_3d_list)
        pred_joints_3d_list = np.array(pred_joints_3d_list)
        pred_left_hand_joint_3d_list = np.array(pred_left_hand_joint_3d_list)
        pred_right_hand_joint_3d_list = np.array(pred_right_hand_joint_3d_list)
        # convert from model format to mo2cap2 format
        original_gt_joint_3d_list = copy.deepcopy(gt_joints_3d_list)
        print(pred_joints_3d_list.shape)

        data_dict = {

        }
        # split by the id name and seq name
        for i in range(len(image_file_path_list)):
            image_file_path = image_file_path_list[i]
            human_name = image_file_path.split('/')[-5] + '_' + image_file_path.split('/')[-4]
            seq_name = image_file_path.split('/')[-3]
            name = human_name + '_' + seq_name
            image_name = image_file_path.split('/')[-1]
            if name not in data_dict.keys():
                data_dict[name] = []

            pred_joints_3d = pred_joints_3d_list[i]
            pred_left_hand_joint_3d = pred_left_hand_joint_3d_list[i]
            pred_right_hand_joint_3d = pred_right_hand_joint_3d_list[i]
            data_dict[name].append({
                'human_name': human_name,
                'seq_name': seq_name,
                'image_name': image_name,
                'image_path': image_file_path,
                'pred_joints_3d': pred_joints_3d,
                'pred_left_hand_joint_3d': pred_left_hand_joint_3d,
                'pred_right_hand_joint_3d': pred_right_hand_joint_3d
            })
        return data_dict

    def load_annotation(self):
        data = self.load_network_pred_pkl(self.egopw_pred_data_pkl_path)
        # convert estimated pose from local pose to global pose
        for name in data.keys():
            for i, item in enumerate(data[name]):
                image_path = item['image_path']
                pred_joints_3d = item['pred_joints_3d']
                pred_left_hand_joint_3d = item['pred_left_hand_joint_3d']
                pred_right_hand_joint_3d = item['pred_right_hand_joint_3d']

                pred_left_wrist = pred_joints_3d[6: 7]
                pred_right_wrist = pred_joints_3d[3: 4]
                ego_pred_left_hand_joint_3d = pred_left_hand_joint_3d + pred_left_wrist - pred_left_hand_joint_3d[0: 1]
                ego_pred_right_hand_joint_3d = pred_right_hand_joint_3d + pred_right_wrist - pred_right_hand_joint_3d[0: 1]

                data[name][i]['image_path'] = image_path
                data[name][i]['image_name'] = item['image_name']
                data[name][i]['human_name'] = item['human_name']
                data[name][i]['seq_name'] = item['seq_name']
                data[name][i]['name'] = name
                data[name][i]['ego_pred_joints_3d'] = pred_joints_3d
                data[name][i]['ego_pred_left_hand_joint_3d'] = ego_pred_left_hand_joint_3d
                data[name][i]['ego_pred_right_hand_joint_3d'] = ego_pred_right_hand_joint_3d

        data_out = []
        if self.split_sequence:
            # split sequence
            for seq_name in data.keys():
                data_seq = data[seq_name]
                data_seq = self.split_into_sequences(data_seq)
                data_out.extend(data_seq)
        else:
            for seq_name in data.keys():
                data_seq = data[seq_name]
                data_out.append(data_seq)
        return data_out

    def split_into_sequences(self, seq):
        seq_len = self.seq_len
        skip_frames = self.skip_frames
        result = []
        for i in range(0, len(seq) - seq_len + 1, skip_frames):
            result.append(seq[i:i + seq_len])
        return result

    def visualize(self, smplx_body_joints):
        # todo: visualization for debug
        # convert smplx to mo2cap2 and hands
        mo2cap2_joints = np.zeros((15, 3))
        mo2cap2_joints[self.mo2cap2_idxs] = smplx_body_joints[self.smplx_idxs_mo2cap2]
        left_hand_joints = np.zeros((21, 3))
        right_hand_joints = np.zeros((21, 3))
        left_hand_joints[self.mano_left_idxs] = smplx_body_joints[self.smplx_idxs_mano_left]
        right_hand_joints[self.mano_right_idxs] = smplx_body_joints[self.smplx_idxs_mano_right]
        from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
        from mmpose.utils.visualization.draw import draw_skeleton_with_chain
        mo2cap2_joints_mesh = draw_skeleton_with_chain(mo2cap2_joints, mo2cap2_chain)
        coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        from mmpose.data.keypoints_mapping.mano import mano_skeleton
        left_hand_joints_mesh = draw_skeleton_with_chain(left_hand_joints, mano_skeleton, keypoint_radius=0.01,
                                                          line_radius=0.0025)
        right_hand_joints_mesh = draw_skeleton_with_chain(right_hand_joints, mano_skeleton, keypoint_radius=0.01,
                                                         line_radius=0.0025)
        open3d.visualization.draw_geometries([mo2cap2_joints_mesh, coord, left_hand_joints_mesh, right_hand_joints_mesh])


    def prepare_data(self, idx):
        """Get data sample."""
        data_idx = self.data[idx]
        len_data_idx = len(data_idx)
        image_path = [data_idx[i]['image_path'] for i in range(len_data_idx)]
        seq_name = [data_idx[i]['seq_name'] for i in range(len_data_idx)]
        name = [data_idx[i]['name'] for i in range(len_data_idx)]
        human_name = [data_idx[i]['human_name'] for i in range(len_data_idx)]
        image_name = [data_idx[i]['image_name'] for i in range(len_data_idx)]
        # load data
        ego_pred_joints_3d = [data_idx[i]['ego_pred_joints_3d'] for i in range(len_data_idx)]
        ego_pred_left_hand_joint_3d = [data_idx[i]['ego_pred_left_hand_joint_3d'] for i in range(len_data_idx)]
        ego_pred_right_hand_joint_3d = [data_idx[i]['ego_pred_right_hand_joint_3d'] for i in range(len_data_idx)]

        # gt_joints_3d_list = [data_idx[i]['gt_joints_3d'] for i in range(len_data_idx)]
        # gt_joints_3d_list = np.asarray(gt_joints_3d_list)
        # ego_camera_pose_list = [data_idx[i]['ego_camera_pose'] for i in range(len_data_idx)]
        # ego_camera_pose_list = np.asarray(ego_camera_pose_list)

        ego_pred_joints_3d = np.asarray(ego_pred_joints_3d)
        ego_pred_left_hand_joint_3d = np.asarray(ego_pred_left_hand_joint_3d)
        ego_pred_right_hand_joint_3d = np.asarray(ego_pred_right_hand_joint_3d)

        ego_smplx_body_joints = np.zeros((len_data_idx, 145, 3))
        ego_smplx_body_joints[:, self.smplx_idxs_mo2cap2] = ego_pred_joints_3d[:, self.mo2cap2_idxs]

        ego_smplx_body_joints[:, self.smplx_idxs_mano_left] = ego_pred_left_hand_joint_3d[:, self.mano_left_idxs]
        ego_smplx_body_joints[:, self.smplx_idxs_mano_right] = ego_pred_right_hand_joint_3d[:, self.mano_right_idxs]

        # set joint center for mo2cap2 -> smplx
        ego_smplx_body_joints[:, 0] = (ego_smplx_body_joints[:, 1] + ego_smplx_body_joints[:, 2]) / 2.0

        # self.visualize(ego_smplx_body_joints[0])

        # global_pred_joints_3d = [data_idx[i]['global_pred_joints_3d'] for i in range(len_data_idx)]
        # global_pred_left_hand_joint_3d = [data_idx[i]['global_pred_left_hand_joint_3d'] for i in range(len_data_idx)]
        # global_pred_right_hand_joint_3d = [data_idx[i]['global_pred_right_hand_joint_3d'] for i in range(len_data_idx)]
        #
        # global_pred_joints_3d = np.asarray(global_pred_joints_3d)
        # global_pred_left_hand_joint_3d = np.asarray(global_pred_left_hand_joint_3d)
        # global_pred_right_hand_joint_3d = np.asarray(global_pred_right_hand_joint_3d)
        #
        # global_smplx_body_joints = np.zeros((len_data_idx, 145, 3))
        # global_smplx_body_joints[:, self.smplx_idxs_mo2cap2] = global_pred_joints_3d[:, self.mo2cap2_idxs]
        # global_smplx_body_joints[:, self.smplx_idxs_mano_left] = global_pred_left_hand_joint_3d[:, self.mano_left_idxs]
        # global_smplx_body_joints[:, self.smplx_idxs_mano_right] = global_pred_right_hand_joint_3d[:, self.mano_right_idxs]
        #
        # global_smplx_body_joints[:, 0] = (global_smplx_body_joints[:, 1] + global_smplx_body_joints[:, 2]) / 2.0

        result_dict = {
            'ego_smplx_joints': ego_smplx_body_joints,
            # 'gt_joints_3d': gt_joints_3d_list,
            'image_path': image_path,
            'seq_name': seq_name,
            'human_name': human_name,
            'name': name,
            'image_name': image_name,
        }

        return result_dict

    def __getitem__(self, idx):
        """Get a sample with given index."""
        # print(f'get item {idx}')
        results = copy.deepcopy(self.prepare_data(idx))
        return self.pipeline(results)

    def evaluate(self, outputs, res_folder, metric=None, logger=None):
        # just save the outputs
        save_path = os.path.join(res_folder, f'outputs.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(outputs, f)
        evaluation_results = {'mpjpe': 0}
        return evaluation_results