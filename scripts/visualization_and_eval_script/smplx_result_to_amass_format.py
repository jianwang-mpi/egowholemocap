#  Copyright Jian Wang @ MPI-INF (c) 2023.
import os
import pickle

import numpy as np
import torch

def smplx_result_to_amass_format(smplx_result_pkl_path, save_path, seg_num=0):
    with open(smplx_result_pkl_path, 'rb') as f:
        data = pickle.load(f)
    # just use data[0] to generate the amass format
    smplx_seq = data[seg_num]['smplx_param']
    # combine the body, hand and global orient
    seq_len = len(smplx_seq['global_orient'])
    poses = torch.concatenate([smplx_seq['global_orient'], smplx_seq['body_pose'],
                               torch.zeros(seq_len, 3 * 3),
                               smplx_seq['left_hand_pose'], smplx_seq['right_hand_pose']], dim=1)
    beta_average = torch.mean(smplx_seq['betas'], dim=0, keepdim=False)
    amass_format = {
        'gender': 'neutral',
        'mocap_frame_rate': 25,
        'num_betas': 16,
        'poses': poses.numpy(),
        'trans': smplx_seq['transl'].numpy(),
        'betas': beta_average.numpy(),
    }
    np.savez(save_path, **amass_format)

if __name__ == '__main__':
    seg_num = 48
    if os.name == 'nt':
        smplx_result_pkl_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_b_256\results_smplx_ik_combined.pkl'
        save_path = fr'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_b_256\results_smplx_ik_combined_amass_format_seg_{seg_num}.npz'
    else:
        smplx_result_pkl_path = r'/CT/EgoMocap/work/EgocentricFullBody/work_dirs/egofullbody_test_b_256/results_smplx_ik_combined.pkl'
        save_path = fr'/CT/EgoMocap/work/EgocentricFullBody/work_dirs/egofullbody_test_b_256/results_smplx_ik_combined_amass_format_seg_{seg_num}.npz'

    smplx_result_to_amass_format(smplx_result_pkl_path, save_path, seg_num=seg_num)