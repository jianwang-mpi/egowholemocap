# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import pickle

import numpy as np
import open3d
import torch
import torch.nn.functional as F
from mmpose.models import POSENETS
from mmpose.models.detectors.base import BasePose
from .diffusion_models.diffusion_refine_pose import GaussianDiffusionRefinePose
from .diffusion_models.model import MotionCondDenoiser
from ..diffusion_mdm.data_loaders.humanml.common.quaternion import qmul, qinv, qrot, euler_to_quaternion
from ...data.keypoints_mapping.mano import mano_skeleton
from ...data.keypoints_mapping.mo2cap2 import mo2cap2_chain
from ...utils.visualization.draw import draw_skeleton_with_chain


class RefinerJointPosition:
    def __init__(self, joint_3d_seq, joint_3d_seq_confidence, mean, std,
                 k=1, total_timesteps=1000, visualize=False):
        self.joint_3d_seq = joint_3d_seq
        self.joint_3d_seq_confidence = joint_3d_seq_confidence
        self.k = k
        self.total_timesteps = total_timesteps

        self.mean = mean
        self.std = std

        self.visualize = visualize
    def get_global_root_rot_trans(self, local_root_velocity, local_root_rot_velocity):
        init_root_quat = torch.asarray([1, 0, 0, 0])
        init_root_trans = torch.asarray([0, 0, 0])
        # recover the root rotation for each frame
        seq_len = len(local_root_rot_velocity)
        root_quat = torch.zeros((seq_len + 1, 4))
        root_trans = torch.zeros((seq_len + 1, 3))
        root_quat[0] = init_root_quat
        root_trans[0] = init_root_trans
        root_quat = root_quat.to(local_root_rot_velocity.device).float()
        root_trans = root_trans.to(local_root_rot_velocity.device).float()
        for i in range(seq_len):
            root_quat[i + 1] = qmul(local_root_rot_velocity[i], root_quat[i])
            root_trans[i + 1] = qrot(qinv(root_quat[i]), local_root_velocity[i]) + root_trans[i]
        return -root_trans, root_quat

    def recover_3D_pose_sequence(self, pose_feature_input):
        # recover with mean and std
        assert len(pose_feature_input) == 1
        pose_feature_input = pose_feature_input[0]
        pose_feature_input = pose_feature_input * self.std + self.mean
        # print(pose_feature_input.shape)
        root_trans_vel = pose_feature_input[:, :2]
        root_rot_vel = pose_feature_input[:, 2:3]
        # recover the trans vel and rot vel
        root_trans_vel_full = torch.zeros((pose_feature_input.shape[0], 3)).float().to(pose_feature_input.device)
        root_trans_vel_full[:, 0] = root_trans_vel[:, 0]
        root_trans_vel_full[:, 2] = root_trans_vel[:, 1]
        root_trans_vel_full = root_trans_vel_full.detach()
        local_root_rotation_velocity_y = np.zeros((pose_feature_input.shape[0], 3))
        local_root_rotation_velocity_y[:, 1] = 1
        local_root_rotation_velocity_y = local_root_rotation_velocity_y * root_rot_vel.detach().cpu().numpy()
        local_root_rotation_velocity_y_recon = euler_to_quaternion(local_root_rotation_velocity_y, order='xyz')
        local_root_rotation_velocity_y_recon = torch.from_numpy(local_root_rotation_velocity_y_recon).float().to(
            pose_feature_input.device)

        pose_info = pose_feature_input[:, 3:]
        pose_info = pose_info.view(-1, 15 + 21 + 21, 3)

        root_trans_vel_processed, root_rot_vel_processed = self.get_global_root_rot_trans(root_trans_vel_full,
                                                                                          local_root_rotation_velocity_y_recon)
        rot_quat_inv = qinv(root_rot_vel_processed)
        data_preprocess_rot_quat_inv_list = torch.repeat_interleave(rot_quat_inv[:, None, :],
                                                                    pose_info.shape[1], dim=1)
        combined_motion = qrot(data_preprocess_rot_quat_inv_list[:-1], pose_info)
        if len(root_trans_vel_processed.shape) == 2:
            root_trans_vel_processed = root_trans_vel_processed[:, None, :]
        combined_motion = combined_motion - root_trans_vel_processed[:-1]

        # visualize the combined motion
        if self.visualize:
            combined_motion_vis = combined_motion.detach().cpu().numpy()
            mo2cap2_pred_motion = combined_motion_vis[:, :15]
            left_hand_pred_motion = combined_motion_vis[:, 15: 15 + 21]
            right_hand_pred_motion = combined_motion_vis[:, 15 + 21:]
            for i in range(0, 196, 10):
                mo2cap2_mesh = draw_skeleton_with_chain(mo2cap2_pred_motion[i], mo2cap2_chain)
                left_hand_mesh = draw_skeleton_with_chain(left_hand_pred_motion[i], mano_skeleton, keypoint_radius=0.01,
                                                                  line_radius=0.0025)
                right_hand_mesh = draw_skeleton_with_chain(right_hand_pred_motion[i], mano_skeleton, keypoint_radius=0.01,
                                                                  line_radius=0.0025)
                coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
                open3d.visualization.draw_geometries([mo2cap2_mesh, left_hand_mesh, right_hand_mesh, coord])

        return combined_motion

    def guided_forward(self, pose_input, diffusion_timestep):
        diffusion_timestep = diffusion_timestep[0].item()

        joint_3d_weight = 1 / (1 + torch.exp(-self.k * (diffusion_timestep - self.total_timesteps * (1 - self.joint_3d_seq_confidence))))
        pose_input = pose_input + (self.joint_3d_seq - pose_input) * joint_3d_weight
        pose_input = pose_input.detach()
        pose_input.requires_grad = True

        recovered_joints_3d = self.recover_3D_pose_sequence(pose_input)
        left_feet_joint = recovered_joints_3d[:, 9]
        right_feet_joint = recovered_joints_3d[:, 13]
        left_feet_velocity = left_feet_joint[1:] - left_feet_joint[:-1]
        left_feet_velocity = torch.norm(left_feet_velocity, dim=-1)
        right_feet_velocity = right_feet_joint[1:] - right_feet_joint[:-1]
        right_feet_velocity = torch.norm(right_feet_velocity, dim=-1)
        # add feet sliding error
        feet_sliding_weight = 6000
        velocities = torch.stack([left_feet_velocity, right_feet_velocity], dim=0)
        min_velocity, min_indices = torch.min(velocities, dim=0)
        velocity_loss = feet_sliding_weight * torch.mean(min_velocity ** 2)
        velocity_loss.backward()
        pose_input = pose_input - pose_input.grad

        return pose_input

@POSENETS.register_module()
class RefineEdgeDiffusionHandsWithRoot(BasePose):

    def __init__(self, representation_dim=(21 + 21 + 15) * 3,
                 cond_feature_dim=15 * 3,
                 guidance_weight=0,
                 cond_drop_prob=1,  # by default always drop the condition
                 seq_len=196,
                 human_body_joint_loss_weight=1.0,
                 mean_std_path=None,
                 visualize=False,
                 ):
        super(RefineEdgeDiffusionHandsWithRoot, self).__init__()

        self.model = MotionCondDenoiser(
            nfeats=representation_dim,
            seq_len=seq_len,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=cond_feature_dim,
            activation=F.gelu,
        )
        self.diffusion_model = GaussianDiffusionRefinePose(
            self.model,
            seq_len,
            representation_dim,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=cond_drop_prob,
            guidance_weight=guidance_weight,
            clip_denoised=False,
            human_body_joint_loss_weight=human_body_joint_loss_weight,
        )


        self.seq_len = seq_len
        with open(mean_std_path, 'rb') as f:
            mean_std = pickle.load(f)
        # combined mean
        self.mean = np.concatenate([mean_std['root_features_mean'],
                                    mean_std['mo2cap2_body_mean'],
                                    mean_std['left_hand_mean'],
                                    mean_std['right_hand_mean']])
        self.std = np.concatenate([mean_std['root_features_std'],
                                      mean_std['mo2cap2_body_std'],
                                        mean_std['left_hand_std'],
                                        mean_std['right_hand_std']])
        self.mean = torch.from_numpy(self.mean).float()
        self.std = torch.from_numpy(self.std).float()
        self.visualize = visualize

    def forward_train(self, img, img_metas, **kwargs):
        pass

    def forward_test(self, img, img_metas, **kwargs):
        pass


    def forward(self, root_features, mo2cap2_body_features, left_hand_features, right_hand_features,
                img_metas=None, return_loss=True, **kwargs):

        # combine all features
        features_all = torch.cat([root_features,
                                  mo2cap2_body_features, left_hand_features, right_hand_features], dim=-1)
        sample_shape = features_all.shape

        # joint_3d_seq_confidence = torch.ones_like(features_all).to(features_all.device) * 0.994
        # joint_3d_seq_confidence[:, :, 7 * 3: 15 * 3] = 0.992
        # joint_3d_seq_confidence[:, :, 15 * 3:] = 0.996
        joint_3d_seq_confidence = torch.ones_like(features_all).to(features_all.device)
        joint_3d_seq_confidence[:, :, 3: 3 + 8 * 3] = 0.999
        joint_3d_seq_confidence[:, :, 3 + 8 * 3: 3 + 11 * 3] = 0.95
        joint_3d_seq_confidence[:, :, 3 + 12 * 3: 3 + 15 * 3] = 0.95
        joint_3d_seq_confidence[:, :, 3 + 15 * 3:] = 0.995
        print(joint_3d_seq_confidence[0])
        joint_position_refiner = RefinerJointPosition(joint_3d_seq=features_all,
                                                      joint_3d_seq_confidence=joint_3d_seq_confidence,
                                                      mean=self.mean.to(features_all.device),
                                                      std=self.std.to(features_all.device),
                                                      visualize=self.visualize)
        sample = self.diffusion_model.p_sample_loop(sample_shape,
                                                    cond=mo2cap2_body_features,
                                                    return_diffusion=False,
                                                    refiner=joint_position_refiner,
                                                    start_point=100)
        result_dict = {'sample': sample,
                       'full_body_features': features_all.detach().cpu(),
                       'img_metas': img_metas}
        return result_dict

    def show_result(self, **kwargs):
        pass

