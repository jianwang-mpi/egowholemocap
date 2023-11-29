#  Copyright Jian Wang @ MPI-INF (c) 2023.
import os
import pickle

import numpy as np
import open3d
import torch
import torch.nn as nn
from colour import Color
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.ik_engine import IK_Engine

from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_chain
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
from mmpose.utils.visualization.draw import draw_skeleton_with_chain


class SMPLXBodyKeyPoints(nn.Module):
    def __init__(self,
                 bm_fname,
                 num_betas=16,
                 n_joints=16,
                 ):
        super(SMPLXBodyKeyPoints, self).__init__()
        self.bm = BodyModel(bm_fname, num_betas=num_betas, persistant_buffer=False)
        self.bm_f = []  # self.bm.f
        self.kpts_colors = np.array([Color('grey').rgb for _ in range(n_joints)])
        self.n_joints = n_joints

    def forward(self, body_parms):
        new_body = self.bm(**body_parms)

        source_kpts_smplx = new_body.Jtr[:, :self.n_joints, :]

        return {'source_kpts': source_kpts_smplx, 'body': new_body}


class SMPLX_IK:
    def __init__(self, keypoint_model=None):
        if os.name == 'nt':
            support_dir = r'Z:\EgoMocap\work\EgocentricFullBody\3rdparty\human_body_prior\support_data\downloads'
        else:
            support_dir = '/CT/EgoMocap/work/EgocentricFullBody/3rdparty/human_body_prior/support_data/downloads'
        vposer_expr_dir = os.path.join(support_dir, 'V02_05')
        self.bm_fname = os.path.join(support_dir, 'models_lockedhead/smplx/SMPLX_NEUTRAL.npz')

        self.comp_device = torch.device('cuda')

        data_loss = torch.nn.MSELoss(reduction='sum')

        stepwise_weights = [{'data': 10., 'poZ_body': 0.0000001, 'betas': .0005}]

        optimizer_args = {'type': 'LBFGS', 'max_iter': 500, 'lr': 2, 'tolerance_change': 1e-5, 'history_size': 300}
        self.ik_engine = IK_Engine(vposer_expr_dir=vposer_expr_dir,
                                   verbosity=2,
                                   display_rc=(2, 2),
                                   data_loss=data_loss,
                                   stepwise_weights=stepwise_weights,
                                   optimizer_args=optimizer_args).to(self.comp_device)
        if keypoint_model is None:
            self.source_pts = SMPLXBodyKeyPoints(bm_fname=self.bm_fname, num_betas=16,
                                                 n_joints=22).to(self.comp_device)
        else:
            self.source_pts = keypoint_model.to(self.comp_device)

    def ik(self, pose_batch: torch.Tensor):

        target_pts = pose_batch.float().to(self.comp_device)

        ik_res = self.ik_engine(self.source_pts, target_pts)
        ik_res_detached = {k: v.detach().cpu() for k, v in ik_res.items()}
        ik_res_detached['target_pts'] = target_pts

        return ik_res_detached


def smplx_result_to_amass_format(smplx_result_dict, save_path):
    # combine the body, hand and global orient
    seq_len = len(smplx_result_dict['root_orient'])
    poses = torch.concatenate([smplx_result_dict['root_orient'], smplx_result_dict['pose_body'],
                               torch.zeros(seq_len, 3 * 3).float(), torch.zeros(seq_len, 15 * 3).float(),
                               torch.zeros(seq_len, 15 * 3).float()],
                              dim=1)
    beta_average = torch.mean(smplx_result_dict['betas'], dim=0, keepdim=False)
    amass_format = {
        'gender': 'neutral',
        'mocap_frame_rate': 25,
        'num_betas': 16,
        'poses': poses.numpy(),
        'trans': smplx_result_dict['trans'].numpy(),
        'betas': beta_average.numpy(),
    }
    np.savez(save_path, **amass_format)


def fit_smplx(mdm_result_pkl, save_path=None):
    with open(mdm_result_pkl, 'rb') as f:
        data = pickle.load(f)
    input_motions = data['input_motions']
    result_motions = data['result_motions']
    print('input_motions shape: ', input_motions.shape)
    assert len(result_motions) == 1
    result_motions = result_motions[0]
    result_motion_tensor = torch.from_numpy(result_motions).float()

    smplx_ik = SMPLX_IK(keypoint_model=None)
    ik_results = smplx_ik.ik(result_motion_tensor)
    print(ik_results.keys())

    if save_path is not None:
        smplx_result_to_amass_format(ik_results, save_path)


def save_skeleton_seq(keypoints_seq, save_dir):
    mo2cap2_keyps_idxs, smplx_keyps_idxs = dset_to_body_model(model_type='smplx', dset='mo2cap2')
    mo2cap2_motion = np.zeros((keypoints_seq.shape[0], 15, 3))
    mo2cap2_motion[:, mo2cap2_keyps_idxs] = keypoints_seq[:, smplx_keyps_idxs]
    for i, mo2cap2_pose in enumerate(mo2cap2_motion):
        pose_mesh = draw_skeleton_with_chain(mo2cap2_pose, mo2cap2_chain)
        if save_dir is not None:
            open3d.io.write_triangle_mesh(os.path.join(save_dir, 'result_%06d.ply' % i), pose_mesh)
        # open3d.visualization.draw_geometries([pose_mesh])


def save_skeleton_result(mdm_result_pkl, save_dir=None):
    with open(mdm_result_pkl, 'rb') as f:
        data = pickle.load(f)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    input_motions = data['input_motions']
    result_motions = data['result_motions']
    print('input_motions shape: ', input_motions.shape)
    assert len(result_motions) == 1
    result_motions = result_motions[0]
    input_motions = input_motions[0]

    result_save_dir = os.path.join(save_dir, 'result')
    input_save_dir = os.path.join(save_dir, 'input')
    os.makedirs(result_save_dir, exist_ok=True)
    os.makedirs(input_save_dir, exist_ok=True)

    save_skeleton_seq(result_motions, result_save_dir)
    save_skeleton_seq(input_motions, input_save_dir)


if __name__ == '__main__':
    if os.name == 'nt':
        mdm_result_path = r'Z:\EgoMocap\work\EgocentricFullBody\vis_results\diffusion_results\47\result.pkl'
        smplx_save_path = r'Z:\EgoMocap\work\EgocentricFullBody\vis_results\diffusion_results\47\result_smplx_ik.npz'
        skeleton_mesh_save_path = r'Z:\EgoMocap\work\EgocentricFullBody\vis_results\diffusion_results\47\skeleton_mesh'
    else:
        mdm_result_path = r'/CT/EgoMocap/work/EgocentricFullBody/vis_results/diffusion_results/47/result.pkl'
        smplx_save_path = r'/CT/EgoMocap/work/EgocentricFullBody/vis_results/diffusion_results/47/result_smplx_ik.npz'
        skeleton_mesh_save_path = r'/CT/EgoMocap/work/EgocentricFullBody/vis_results/diffusion_results/47/skeleton_mesh'
    # fit_smplx(mdm_result_pkl=mdm_result_path,
    #           save_path=save_path)
    save_skeleton_result(mdm_result_pkl=mdm_result_path,
                         save_dir=skeleton_mesh_save_path)
