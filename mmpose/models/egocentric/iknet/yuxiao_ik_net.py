#  Copyright Jian Wang @ MPI-INF (c) 2023.

import torch.nn as nn
import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core import imshow_bboxes, imshow_keypoints
from mmpose.models import builder
from mmpose.models.builder import POSENETS
from mmpose.models.detectors.base import BasePose
import torch
import smplx


def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

@POSENETS.register_module()
class SimpleIkNet(nn.Module):
    def __init__(self, feat_dim=1024, joint_num=15, center_pose=True,
                 norm_pose=False, smplx_config=None):
        """

        Args:
            feat_dim: input feature dimension
            joint_num: input 3d joint number
            smplx_config: smplx model setting, dict containing necessary information for initializing smplx model
        """

        super(SimpleIkNet, self).__init__()

        if smplx_config is not None:
            self.smplx = smplx.create(**smplx_config)
        else:
            # use default setting
            smplx_config = dict(
                model_path='/CT/EgoMocap/work/smpl_models/models_smplx_v1_1/models/SMPLX_NEUTRAL.npz',
                model_type='smplx', use_pca=False, flat_hand_mean=True, num_betas=10
            )
            self.smplx = smplx.create(**smplx_config)

        self.joint_num = joint_num
        self.body_conv = make_linear_layers([feat_dim, 512], relu_final=False)
        self.root_pose_out = make_linear_layers([self.joint_num * 3 + 512, 6], relu_final=False)
        self.body_pose_out = make_linear_layers(
            [512+3 * self.joint_num, 21 * 6], relu_final=False, use_bn=False)  # without root
        self.shape_out = make_linear_layers([feat_dim, smplx_config['num_betas']], relu_final=False,
                                            use_bn=False)
        self.global_transl_out = make_linear_layers([feat_dim, 3], relu_final=False, use_bn=False)
        self.feat_dim = feat_dim

    def forward(self, image_features, keypoints_3d):
        batch_size = image_features.shape[0]

        # shape parameter
        shape_param = self.shape_out(image_features)

        # body pose parameter
        body_pose_token = self.body_conv(body_pose_token)
        body_pose_token = torch.cat((body_pose_token, body_joint_img), 2)
        root_pose = self.root_pose_out(body_pose_token.view(batch_size, -1))
        body_pose = self.body_pose_out(body_pose_token.view(batch_size, -1))

        return root_pose, body_pose, shape_param, cam_param