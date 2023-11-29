#  Copyright Jian Wang @ MPI-INF (c) 2023.
import copy
import pickle

import numpy as np
import open3d
from torch.utils.data import Dataset

from mmpose.data.keypoints_mapping.hml import hml_joint_names
from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_joint_names
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.diffusion.mo2cap2_keypoints_to_hml3d import KeypointsToHumanML3D
from mmpose.datasets.pipelines import Compose


@DATASETS.register_module()
class Mo2Cap2MotionDataset(Dataset):
    def __init__(self, path_dict,
                 frame_rate,
                 seq_len,
                 mean_path,
                 std_path,
                 pipeline,
                 test_mode=False):
        self.path_dict = path_dict
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        self.output_frame_rate = frame_rate
        self.seq_len = seq_len
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

        self.dst_idxs, self.model_idxs = self.dset_to_body_model(dset='mo2cap2', model_type='hml')

        self.keypoints_seq, self.keypoints_gt_seq, self.img_name_seqs = self.load_annotation()
        # remove the last pose in each sequence

        self.keypoints_to_hml3d_class = KeypointsToHumanML3D()
        self.data = self.keypoints_to_hml3d(self.keypoints_seq)
        self.data = self.split_into_sequences(self.data, self.keypoints_gt_seq, self.img_name_seqs)
        self.data = self.normalize_data(self.data)
        self.data = self.nan_to_zero(self.data)


    def normalize_data(self, data):
        for i in range(len(data)):
            data[i]['data'] = (data[i]['data'] - self.mean) / self.std
        return data

    def nan_to_zero(self, data):
        for i in range(len(data)):
            data[i]['data'] = np.nan_to_num(data[i]['data'])
        return data

    def __len__(self):
        return len(self.data)

    def keypoints_to_hml3d(self, keypoints_seq_list):
        hml3d_seq_list = []
        for keypoints_seq in keypoints_seq_list:
            data, ground_positions, positions, l_velocity = self.keypoints_to_hml3d_class.process_file(
                keypoints_seq, 0.002)

            hml3d_seq_list.append(data)
        return hml3d_seq_list

    def load_path_dict(self):
        joint_location_seqs = []
        for seq_name in self.path_dict.keys():
            # load joint sequence here and convert from mo2cap2 representation to the one used in humanml\
            motion_path = self.path_dict[seq_name]
            with open(motion_path, 'rb') as f:
                mo2cap2_pose_seq = pickle.load(f)

            seq_len, joint_num, _ = mo2cap2_pose_seq.shape
            assert joint_num == 15

            # convert it to hml representation
            hml_joints = np.zeros((seq_len, 22, 3))
            hml_joints[:, self.model_idxs] = mo2cap2_pose_seq[:, self.dst_idxs]
            hml_joints[:, 0] = (hml_joints[:, 1] + hml_joints[:, 2]) / 2

            # hml_joints = np.dot(hml_joints, trans_matrix)
            # hml_joints[..., 0] *= -1

            # visualize hml joints
            # from mmpose.utils.visualization.draw import draw_keypoints_3d
            # mesh = draw_keypoints_3d(hml_joints[0])
            # coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
            # open3d.visualization.draw_geometries([mesh, coord])

            joint_location_seqs.append(hml_joints)
        return joint_location_seqs

    def load_annotation(self):
        # trans_matrix = np.array([[1.0, 0.0, 0.0],
        #                          [0.0, 0.0, 1.0],
        #                          [0.0, 1.0, 0.0]])

        if isinstance(self.path_dict, dict):
            return self.load_path_dict()
        elif isinstance(self.path_dict, str):
            with open(self.path_dict, 'rb') as f:
                data_seq_list = pickle.load(f)
            joint_location_seqs = []
            img_name_seqs = []
            keypoints_gt_seq = []
            for seq_name in data_seq_list.keys():
                data_seq = data_seq_list[seq_name]
                mo2cap2_keypoints_seq = np.empty((len(data_seq), 15, 3))
                mo2cap2_keypoints_gt_seq = np.empty((len(data_seq), 15, 3))
                img_name_seq = [None] * len(data_seq)
                for i in range(len(data_seq)):
                    mo2cap2_keypoints_seq[i] = data_seq[i]['global_pose_pred']
                    mo2cap2_keypoints_gt_seq[i] = data_seq[i]['global_pose_gt']
                    img_name_seq[i] = data_seq[i]['image_file']

                seq_len, joint_num, _ = mo2cap2_keypoints_seq.shape
                assert joint_num == 15

                # convert it to hml representation
                hml_joints = np.zeros((seq_len, 22, 3))
                hml_joints[:, self.model_idxs] = mo2cap2_keypoints_seq[:, self.dst_idxs]
                hml_joints[:, 0] = (hml_joints[:, 1] + hml_joints[:, 2]) / 2


                joint_location_seqs.append(hml_joints)
                keypoints_gt_seq.append(mo2cap2_keypoints_gt_seq)
                img_name_seqs.append(img_name_seq)
            return joint_location_seqs, keypoints_gt_seq, img_name_seqs


    def split_into_sequences(self, seq_list, seq_gt_list, image_names_list):
        result_list = []
        for seq, seq_gt, image_names in zip(seq_list, seq_gt_list, image_names_list):
            # note: crop the first pose in each sequence
            # ---------------------------------
            seq_gt = seq_gt[1:]
            image_names = image_names[1:]

            # --------------------------
            seq_length = len(seq)
            seq_gt_length = len(seq_gt)
            print('seq_length', seq_length)
            print('seq_gt_length', seq_gt_length)
            assert seq_length == seq_gt_length == len(image_names)
            for i in range(0, seq_length, self.seq_len):
                result_seq = seq[i:i + self.seq_len]
                result_seq_gt = seq_gt[i:i + self.seq_len]
                result_image_names = image_names[i:i + self.seq_len]
                mask = np.ones((result_seq.shape[0], result_seq.shape[1], 1), dtype=float)
                current_len_result_seq = len(result_seq)
                if current_len_result_seq < self.seq_len:
                    result_seq = np.concatenate([result_seq,
                                                 np.zeros((self.seq_len - current_len_result_seq, result_seq.shape[1]),
                                                          dtype=float)
                                                 ], axis=0)
                    result_seq_gt = np.concatenate([result_seq_gt,
                                                    np.zeros((self.seq_len - current_len_result_seq,
                                                              result_seq_gt.shape[1], 3), dtype=float)
                                                    ], axis=0)
                    mask = np.concatenate([mask,
                                           np.zeros((self.seq_len - current_len_result_seq, result_seq.shape[1], 1),
                                                    dtype=float)
                                           ], axis=0)
                result_list.append({'data': result_seq, 'mask': mask, 'lengths': np.asarray(current_len_result_seq),
                                    'mean': self.mean, 'std': self.std, 'gt': result_seq_gt,
                                    'image_names': result_image_names})
        return result_list

    def prepare_data(self, idx):
        """Get data sample."""
        result = self.data[idx]
        return result

    def __getitem__(self, idx):
        """Get a sample with given index."""
        # print(f'get item {idx}')
        results = copy.deepcopy(self.prepare_data(idx))
        return self.pipeline(results)

    def dset_to_body_model(self, model_type='hml', dset='mo2cap2'):

        mapping = {}

        if model_type == 'hml':
            keypoint_names = hml_joint_names
        else:
            raise ValueError('Unknown model dataset: {}'.format(model_type))

        if dset == 'mo2cap2':
            dset_keyp_names = mo2cap2_joint_names
        else:
            raise ValueError('Unknown dset dataset: {}'.format(dset))

        for idx, name in enumerate(keypoint_names):
            if name in dset_keyp_names:
                mapping[idx] = dset_keyp_names.index(name)

        model_keyps_idxs = np.array(list(mapping.keys()), dtype=np.int32)
        dset_keyps_idxs = np.array(list(mapping.values()), dtype=np.int32)

        return dset_keyps_idxs, model_keyps_idxs
