#  Copyright Jian Wang @ MPI-INF (c) 2023.

# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import torch

from mmpose.models import POSENETS
from mmpose.models.detectors.base import BasePose
from .model_egoego.transformer_uncond_diffusion_model import UncondGaussianDiffusion


# from .diffusion_models.diffusion import GaussianDiffusion
# from .diffusion_models.model import MotionCondDenoiser


@POSENETS.register_module()
class EgoEgoDiffusion(BasePose):
    def __init__(self, right_hand=True, load_model_path=None, representation_dim=(15 + 21 + 21) * 3,
                 model_dim=512,
                 num_dec_layers=4,
                 num_head=4,
                 loss_type='l1',
                 seq_len=196,
                 return_diffusion=False):
        super(EgoEgoDiffusion, self).__init__()
        self.load_model_path = load_model_path
        self.right_hand = right_hand
        self.return_diffusion = return_diffusion

        self.diffusion_model = UncondGaussianDiffusion(
            d_feats=representation_dim, d_model=model_dim,
            n_dec_layers=num_dec_layers, n_head=num_head, d_k=256, d_v=256,
            max_timesteps=seq_len + 1, out_dim=representation_dim, timesteps=1000,
            objective="pred_x0", loss_type=loss_type,
        )

        if load_model_path is not None:
            print(f"Loading checkpoints from [{self.model_path}]...")
            state_dict = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(state_dict['model'], strict=True)

        self.seq_len = seq_len

    def forward_train(self, left_hand_features, right_hand_features, mo2cap2_body_features, img_metas, **kwargs):
        input_features = torch.cat([mo2cap2_body_features, left_hand_features, right_hand_features], dim=-1)
        mpjpe_loss = self.diffusion_model(input_features)
        loss_dict = {'mpjpe_loss': mpjpe_loss}
        return loss_dict

    def forward_test(self, left_hand_features, right_hand_features, mo2cap2_body_features, img_metas, **kwargs):
        input_features = torch.cat([mo2cap2_body_features, left_hand_features, right_hand_features], dim=-1)
        sample = self.diffusion_model.sample(input_features)
        result_dict = {'sample': sample,
                       'mo2cap2_body_features': mo2cap2_body_features.detach().cpu(),
                       'img_metas': img_metas}
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
