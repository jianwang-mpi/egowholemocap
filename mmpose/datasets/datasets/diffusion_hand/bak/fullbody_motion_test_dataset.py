#  Copyright Jian Wang @ MPI-INF (c) 2023.
import copy
import os
import pickle

import numpy as np
import open3d
import torch
from torch.utils.data import Dataset

from mmpose.data.keypoints_mapping.hml import hml_joint_names
from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_joint_names
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.diffusion.mo2cap2_keypoints_to_hml3d import KeypointsToHumanML3D
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
from mmpose.datasets.pipelines import Compose


@DATASETS.register_module()
class FullBodyMotionTestDataset(Dataset):
    frame_rate = 25

    dataset_path_dict = {
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

    def __init__(self, data_pkl_path,
                 seq_len,
                 skip_frames,
                 pipeline,
                 output_frame_rate=25,
                 split_sequence=True,
                 test_mode=False):
        self.data_pkl_path = data_pkl_path
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

        ext_id_list = []
        seq_name_list = []
        pred_joints_3d_list = []
        pred_left_hand_joint_3d_list = []
        pred_right_hand_joint_3d_list = []
        gt_joints_3d_list = []
        ego_camera_pose_list = []
        for _, pred in enumerate(data):
            pred_joints_3d_item = pred['body_pose_results']['keypoints_pred']
            pred_left_hand_joint_item = pred['left_hands_preds']['joint_cam_transformed']
            pred_right_hand_joint_item = pred['right_hands_preds']['joint_cam_transformed']
            if torch.is_tensor(pred_joints_3d_item):
                pred_joints_3d_item = pred_joints_3d_item.cpu().numpy()
            if torch.is_tensor(pred_left_hand_joint_item):
                pred_left_hand_joint_item = pred_left_hand_joint_item.cpu().numpy()
                pred_right_hand_joint_item = pred_right_hand_joint_item.cpu().numpy()

            # get the uncertainty
            if 'keypoint_confidence' in pred['body_pose_results'].keys():
                uncertainty_body = pred['body_pose_results']['keypoint_confidence']
            else:
                # use default uncertainty
                uncertainty_body = np.ones_like(pred_joints_3d_item)
                uncertainty_body[8:] *= 0.9995
            if 'keypoint_confidence' in pred['left_hands_preds'].keys():
                uncertainty_lhand = pred['left_hands_preds']['keypoint_confidence']
            else:
                # use default uncertainty
                uncertainty_lhand = np.ones_like(pred_left_hand_joint_item) * 0.9995

            if 'keypoint_confidence' in pred['right_hands_preds'].keys():
                uncertainty_rhand = pred['right_hands_preds']['keypoint_confidence']
            else:
                # use default uncertainty
                uncertainty_rhand = np.ones_like(pred_right_hand_joint_item) * 0.9995

            pred_joints_3d_list.extend(pred_joints_3d_item)
            pred_left_hand_joint_3d_list.extend(pred_left_hand_joint_item)
            pred_right_hand_joint_3d_list.extend(pred_right_hand_joint_item)
            img_meta_list = pred['img_metas']
            for img_meta_item in img_meta_list:
                ext_id = img_meta_item['ext_id']
                seq_name = img_meta_item['seq_name']
                gt_joints_3d_item = img_meta_item['keypoints_3d']
                ego_camera_pose_item = img_meta_item['ego_camera_pose']
                gt_joints_3d_list.append(gt_joints_3d_item)
                ego_camera_pose_list.append(ego_camera_pose_item)
                ext_id_list.append(ext_id)
                seq_name_list.append(seq_name)

        gt_joints_3d_list = np.array(gt_joints_3d_list)
        ego_camera_pose_list = np.array(ego_camera_pose_list)
        pred_joints_3d_list = np.array(pred_joints_3d_list)
        pred_left_hand_joint_3d_list = np.array(pred_left_hand_joint_3d_list)
        pred_right_hand_joint_3d_list = np.array(pred_right_hand_joint_3d_list)

        # split by seq names
        data_by_seq_name = {}
        for i, seq_name in enumerate(seq_name_list):
            if seq_name not in data_by_seq_name.keys():
                data_by_seq_name[seq_name] = []
            ext_id = ext_id_list[i]
            ego_camera_pose = ego_camera_pose_list[i]
            gt_joints_3d = gt_joints_3d_list[i]
            pred_joints_3d = pred_joints_3d_list[i]
            pred_left_hand_joint_3d = pred_left_hand_joint_3d_list[i]
            pred_right_hand_joint_3d = pred_right_hand_joint_3d_list[i]
            data_by_seq_name[seq_name].append({'ext_id': ext_id,
                                                  'ego_camera_pose': ego_camera_pose,
                                                  'gt_joints_3d': gt_joints_3d,
                                                  'pred_joints_3d': pred_joints_3d,
                                                  'pred_left_hand_joint_3d': pred_left_hand_joint_3d,
                                                  'pred_right_hand_joint_3d': pred_right_hand_joint_3d})
        for seq_name in data_by_seq_name.keys():
            data_by_seq_name[seq_name] = sorted(data_by_seq_name[seq_name], key=lambda x: x['ext_id'])

        return data_by_seq_name

    def transform_new_body_pose(self, keypoints, fisheye_camera_transformation_matrix):
        keypoints_homo = np.ones((len(keypoints), 4))
        keypoints_homo[:, :3] = keypoints
        global_keypoints_homo = fisheye_camera_transformation_matrix.dot(keypoints_homo.T).T
        transformed_pose = global_keypoints_homo[:, :3].astype(np.float32)
        return transformed_pose

    def load_annotation(self):
        data = self.load_network_pred_pkl(self.data_pkl_path)
        # convert estimated pose from local pose to global pose
        for seq_name in data.keys():
            for i, item in enumerate(data[seq_name]):
                ext_id = item['ext_id']
                ego_camera_pose = item['ego_camera_pose']
                pred_joints_3d = item['pred_joints_3d']
                pred_left_hand_joint_3d = item['pred_left_hand_joint_3d']
                pred_right_hand_joint_3d = item['pred_right_hand_joint_3d']

                # note:
                # 这里出问题了，预测的mano joint可能不是在egocentric space里面的，这个mano的joint的location其实
                # 是受到wrist location的控制的，所以这里需要先提取出来wrist location，然后对mano joint进行处理
                pred_left_wrist = pred_joints_3d[6: 7]
                pred_right_wrist = pred_joints_3d[3: 4]
                ego_pred_left_hand_joint_3d = pred_left_hand_joint_3d + pred_left_wrist - pred_left_hand_joint_3d[0: 1]
                ego_pred_right_hand_joint_3d = pred_right_hand_joint_3d + pred_right_wrist - pred_right_hand_joint_3d[0: 1]

                data[seq_name][i]['ext_id'] = ext_id
                data[seq_name][i]['seq_name'] = seq_name
                data[seq_name][i]['ego_pred_joints_3d'] = pred_joints_3d
                data[seq_name][i]['ego_pred_left_hand_joint_3d'] = ego_pred_left_hand_joint_3d
                data[seq_name][i]['ego_pred_right_hand_joint_3d'] = ego_pred_right_hand_joint_3d

                transformed_pred_joints_3d = self.transform_new_body_pose(pred_joints_3d, ego_camera_pose)
                transformed_pred_left_hand_joint_3d = self.transform_new_body_pose(ego_pred_left_hand_joint_3d, ego_camera_pose)
                transformed_pred_right_hand_joint_3d = self.transform_new_body_pose(ego_pred_right_hand_joint_3d, ego_camera_pose)
                data[seq_name][i]['global_pred_joints_3d'] = transformed_pred_joints_3d
                data[seq_name][i]['global_pred_left_hand_joint_3d'] = transformed_pred_left_hand_joint_3d
                data[seq_name][i]['global_pred_right_hand_joint_3d'] = transformed_pred_right_hand_joint_3d
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
        ext_id = [data_idx[i]['ext_id'] for i in range(len_data_idx)]
        seq_name = [data_idx[i]['seq_name'] for i in range(len_data_idx)]
        # load data


        ego_pred_joints_3d = [data_idx[i]['ego_pred_joints_3d'] for i in range(len_data_idx)]
        ego_pred_left_hand_joint_3d = [data_idx[i]['ego_pred_left_hand_joint_3d'] for i in range(len_data_idx)]
        ego_pred_right_hand_joint_3d = [data_idx[i]['ego_pred_right_hand_joint_3d'] for i in range(len_data_idx)]

        gt_joints_3d_list = [data_idx[i]['gt_joints_3d'] for i in range(len_data_idx)]
        gt_joints_3d_list = np.asarray(gt_joints_3d_list)
        ego_camera_pose_list = [data_idx[i]['ego_camera_pose'] for i in range(len_data_idx)]
        ego_camera_pose_list = np.asarray(ego_camera_pose_list)

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

        global_pred_joints_3d = [data_idx[i]['global_pred_joints_3d'] for i in range(len_data_idx)]
        global_pred_left_hand_joint_3d = [data_idx[i]['global_pred_left_hand_joint_3d'] for i in range(len_data_idx)]
        global_pred_right_hand_joint_3d = [data_idx[i]['global_pred_right_hand_joint_3d'] for i in range(len_data_idx)]

        global_pred_joints_3d = np.asarray(global_pred_joints_3d)
        global_pred_left_hand_joint_3d = np.asarray(global_pred_left_hand_joint_3d)
        global_pred_right_hand_joint_3d = np.asarray(global_pred_right_hand_joint_3d)

        global_smplx_body_joints = np.zeros((len_data_idx, 145, 3))
        global_smplx_body_joints[:, self.smplx_idxs_mo2cap2] = global_pred_joints_3d[:, self.mo2cap2_idxs]
        global_smplx_body_joints[:, self.smplx_idxs_mano_left] = global_pred_left_hand_joint_3d[:, self.mano_left_idxs]
        global_smplx_body_joints[:, self.smplx_idxs_mano_right] = global_pred_right_hand_joint_3d[:, self.mano_right_idxs]

        global_smplx_body_joints[:, 0] = (global_smplx_body_joints[:, 1] + global_smplx_body_joints[:, 2]) / 2.0

        result_dict = {
            'ego_smplx_joints': ego_smplx_body_joints,
            'global_smplx_joints': global_smplx_body_joints,
            'gt_joints_3d': gt_joints_3d_list,
            'ego_camera_pose': ego_camera_pose_list,
            'ext_id': ext_id,
            'seq_name': seq_name,
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