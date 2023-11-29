#  Copyright Jian Wang @ MPI-INF (c) 2023.

# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import torch
import torch.nn.functional as F

from mmpose.models import POSENETS
from mmpose.models.detectors.base import BasePose
from .diffusion_models.diffusion import GaussianDiffusion
from .diffusion_models.model import MotionCondDenoiser


@POSENETS.register_module()
class DiffusionEDGEHand(BasePose):
    def __init__(self, load_model_path=None,
                 representation_dim=20 * 3,
                 is_left_hand=True,
                 cond_feature_dim=15 * 3,
                 guidance_weight=0,
                 cond_drop_prob=1,  # by default always drop the condition
                 seq_len=196,
                 human_body_joint_loss_weight=1.0,
                 return_diffusion=False):
        super(DiffusionEDGEHand, self).__init__()
        self.load_model_path = load_model_path
        self.return_diffusion = return_diffusion
        self.is_left_hand = is_left_hand

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
        self.diffusion_model = GaussianDiffusion(
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

        if load_model_path is not None:
            print(f"Loading checkpoints from [{self.model_path}]...")
            state_dict = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(state_dict['model'], strict=True)

        self.seq_len = seq_len

    def forward_train(self, left_hand_features, right_hand_features, mo2cap2_body_features,
                      img_metas, **kwargs):

        # combine all features
        if self.is_left_hand:
            features_all = left_hand_features[:, :,  3:]
        else:
            features_all = right_hand_features[:, :, 3:]
        # features_all = torch.cat([mo2cap2_body_features,
        #                           left_hand_features, right_hand_features], dim=-1)
        loss_dict = self.diffusion_model(features_all, cond=mo2cap2_body_features)
        return loss_dict

    def forward_test(self, left_hand_features, right_hand_features, mo2cap2_body_features, img_metas, **kwargs):
        # combine all features
        features_all = left_hand_features[:, :, 3:]
        sample_shape = features_all.shape
        sample, diffusion_list = self.diffusion_model.p_sample_loop(sample_shape, cond=mo2cap2_body_features,
                                                                    return_diffusion=True)
        diffusion_list = [diffusion.detach().cpu().numpy() for diffusion in diffusion_list]
        diffusion_list = diffusion_list[::100]
        result_dict = {'sample': sample,
                       'mo2cap2_body_features': mo2cap2_body_features.detach().cpu(),
                       'img_metas': img_metas}
        if self.return_diffusion:
            result_dict['diffusion_list'] = diffusion_list
        return result_dict

    def forward(self, left_hand_features, right_hand_features, mo2cap2_body_features,
                img_metas=None, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(left_hand_features, right_hand_features, mo2cap2_body_features,
                                      img_metas, **kwargs)
        else:
            return self.forward_test(left_hand_features, right_hand_features, mo2cap2_body_features,
                                     img_metas, **kwargs)

    def show_result(self, **kwargs):
        pass
