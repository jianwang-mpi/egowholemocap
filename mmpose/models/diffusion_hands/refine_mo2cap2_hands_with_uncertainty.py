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
from ..diffusion_mdm.data_loaders.humanml.common.quaternion import qinv, qrot, euler_to_quaternion, qmul
from ...utils.visualization.draw import draw_skeleton_with_chain


class RefinerJointPosition:
    def __init__(self, joint_3d_seq, joint_3d_seq_confidence, k=1, total_timesteps=1000):
        self.joint_3d_seq = joint_3d_seq
        self.joint_3d_seq_confidence = joint_3d_seq_confidence
        self.k = k
        self.total_timesteps = total_timesteps


    def guided_forward(self, pose_input, diffusion_timestep):
        diffusion_timestep = diffusion_timestep[0].item()
        joint_3d_weight = 1 / (1 + torch.exp(
            -self.k * (diffusion_timestep - self.total_timesteps * (1 - self.joint_3d_seq_confidence))))
        pose_input = pose_input + (self.joint_3d_seq - pose_input) * joint_3d_weight
        return pose_input

@POSENETS.register_module()
class RefineEdgeDiffusionHandsUncertainty(BasePose):

    def __init__(self, representation_dim=(21 + 21 + 15) * 3,
                 cond_feature_dim=15 * 3,
                 guidance_weight=0,
                 cond_drop_prob=1,  # by default always drop the condition
                 seq_len=196,
                 human_body_joint_loss_weight=1.0):
        super(RefineEdgeDiffusionHandsUncertainty, self).__init__()

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
            human_body_joint_loss_weight=human_body_joint_loss_weight
        )

        # if load_model_path is not None:
        #     print(f"Loading checkpoints from [{self.model_path}]...")
        #     state_dict = torch.load(self.model_path, map_location='cpu')
        #     self.model.load_state_dict(state_dict['model'], strict=True)

        self.seq_len = seq_len

    def forward_train(self, img, img_metas, **kwargs):
        pass

    def forward_test(self, img, img_metas, **kwargs):
        pass


    def forward(self, mo2cap2_body_features, left_hand_features, right_hand_features, uncertainty,
                img_metas=None, return_loss=True, **kwargs):

        # combine all features
        features_all = torch.cat([mo2cap2_body_features, left_hand_features, right_hand_features], dim=-1)
        sample_shape = features_all.shape

        assert sample_shape == uncertainty.shape
        joint_3d_seq_confidence = uncertainty
        # joint_3d_seq_confidence = torch.ones_like(features_all).to(features_all.device) * 2
        # joint_3d_seq_confidence[:, :, 7 * 3: 15 * 3] = 0.998
        # joint_3d_seq_confidence[:, :, 15 * 3:] = 0.995
        # joint_3d_seq_confidence[:, 30: 50, 15 * 3: (15 + 21) * 3] = 0.998
        # joint_3d_seq_confidence = torch.ones_like(features_all).to(features_all.device) * 1
        # joint_3d_seq_confidence[:, :, 7 * 3: 15 * 3] = 0.998
        # joint_3d_seq_confidence[:, :, 15 * 3:] = 0.998
        joint_position_refiner = RefinerJointPosition(joint_3d_seq=features_all,
                                                      joint_3d_seq_confidence=joint_3d_seq_confidence)
        sample = self.diffusion_model.p_sample_loop(sample_shape, cond=mo2cap2_body_features,
                                                                    return_diffusion=False,
                                                                    refiner=joint_position_refiner,
                                                    start_point=100)
        result_dict = {'sample': sample,
                       'full_body_features': features_all.detach().cpu(),
                       'img_metas': img_metas}
        return result_dict

    def show_result(self, **kwargs):
        pass

