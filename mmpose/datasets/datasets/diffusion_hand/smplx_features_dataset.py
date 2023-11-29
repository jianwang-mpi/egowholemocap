#  Copyright Jian Wang @ MPI-INF (c) 2023.
#  Copyright Jian Wang @ MPI-INF (c) 2023.
import copy
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from mmpose.datasets.builder import DATASETS
from mmpose.datasets.pipelines import Compose


@DATASETS.register_module()
class EgoSMPLXFeaturesDataset(Dataset):
    # directly load the pre-processed features
    def __init__(self, data_path_list,
                 seq_len,
                 skip_frame,
                 pipeline, split_sequence=True, test_mode=False):
        self.data_path_list = data_path_list  # smplx joint path list
        self.skip_frame = skip_frame
        self.seq_len = seq_len

        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.data = self.load_annotation(split_sequence=split_sequence)

        self.eval_times = 0

    def load_annotation(self, split_sequence=True):
        with open(self.data_path_list, 'rb') as f:
            features_seqs_list = pickle.load(f)
        if split_sequence:
            left_hand_features_seqs = features_seqs_list['left_hand_features_seqs']
            right_hand_features_seqs = features_seqs_list['right_hand_features_seqs']
            mo2cap2_body_features_seqs = features_seqs_list['mo2cap2_body_features_seqs']
            if 'root_features_seqs' in features_seqs_list.keys():
                root_features_seqs = features_seqs_list['root_features_seqs']
            img_metas_seqs = features_seqs_list['image_metas_seqs']
            left_hand_features_seq_list = []
            for seq in left_hand_features_seqs:
                left_hand_features_seq_list.extend(self.split_into_sequences(seq, skip=self.skip_frame))
            right_hand_features_seq_list = []
            for seq in right_hand_features_seqs:
                right_hand_features_seq_list.extend(self.split_into_sequences(seq, skip=self.skip_frame))
            mo2cap2_body_features_seq_list = []
            for seq in mo2cap2_body_features_seqs:
                mo2cap2_body_features_seq_list.extend(self.split_into_sequences(seq, skip=self.skip_frame))
            root_features_seq_list = []
            if 'root_features_seqs' in features_seqs_list.keys():
                for seq in root_features_seqs:
                    dummy_seq = copy.deepcopy(seq)
                    dummy_seq = np.concatenate([dummy_seq, np.zeros((1, 3))], axis=0)
                    splitted_seq_list = self.split_into_sequences(dummy_seq, skip=self.skip_frame)
                    # splitted_seq_list = [s[:-1] for s in splitted_seq_list]
                    root_features_seq_list.extend(splitted_seq_list)
            # img_metas_seq_list_dict = {}
            # for img_meta_seq in img_metas_seqs:
            #     for key in img_meta_seq.keys():
            #         if key not in img_metas_seq_list_dict.keys():
            #             img_metas_seq_list_dict[key] = []
            #         item_list = self.split_into_sequences(img_meta_seq[key], skip=self.skip_frame)
            #         img_metas_seq_list_dict[key].extend(item_list)
            global_smplx_joints_seq_list = []
            ego_smplx_joints_seq_list = []
            for img_meta_seq in img_metas_seqs:
                if 'global_smplx_joints' in img_meta_seq.keys():
                    global_smplx_joints_seq_list.extend(self.split_into_sequences(img_meta_seq['global_smplx_joints'],
                                                                               skip=self.skip_frame))
                if 'ego_smplx_joints' in img_meta_seq.keys():
                    ego_smplx_joints_seq_list.extend(self.split_into_sequences(img_meta_seq['ego_smplx_joints'],
                                                                               skip=self.skip_frame))

            data_dict = {
                'left_hand_features': left_hand_features_seq_list,
                'right_hand_features': right_hand_features_seq_list,
                'mo2cap2_body_features': mo2cap2_body_features_seq_list,
                # 'root_features': root_features_seq_list,
                # 'global_smplx_joints': global_smplx_joints_seq_list,
                # 'ego_smplx_joints': ego_smplx_joints_seq_list,
            }
            if 'root_features_seqs' in features_seqs_list.keys():
                data_dict['root_features'] = root_features_seq_list
            if len(global_smplx_joints_seq_list) > 0:
                data_dict['global_smplx_joints'] = global_smplx_joints_seq_list
            if len(ego_smplx_joints_seq_list) > 0:
                data_dict['ego_smplx_joints'] = ego_smplx_joints_seq_list
            data = []
            for i in range(len(data_dict['left_hand_features'])):
                item = {}
                for key in data_dict.keys():
                    item[key] = data_dict[key][i]
                data.append(item)
        else:
            raise Exception('should split the sequence!')
        if self.test_mode is True:
            data = data[:100]
        return data

    def split_into_sequences(self, seq, skip=10, frame_skip=1):
        seq_len = self.seq_len
        result = []
        for i in range(0, len(seq) - seq_len + 1, skip):
            result.append(seq[i: i + seq_len: frame_skip])
        return result

    def __len__(self):
        return len(self.data)

    def prepare_data(self, idx):
        data_i = self.data[idx]
        return data_i

    def evaluate(self, outputs, res_folder, metric=None, logger=None):
        # just save the outputs
        save_path = os.path.join(res_folder, f'outputs_eval_{self.eval_times}.pkl') # stupid way to record the epoch id
        with open(save_path, 'wb') as f:
            pickle.dump(outputs, f)
        self.eval_times += 1
        evaluation_results = {'mpjpe': 0}
        return evaluation_results

    def __getitem__(self, idx):
        """Get a sample with given index."""
        # print(f'get item {idx}')
        results = copy.deepcopy(self.prepare_data(idx))
        return self.pipeline(results)