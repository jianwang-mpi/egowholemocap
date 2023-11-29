#  Copyright Jian Wang @ MPI-INF (c) 2023.

import torch.nn as nn
import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow
from mmpose.models.egocentric.iknet.smplx_layer import SMPLXLayer

from mmpose.core import imshow_bboxes, imshow_keypoints
from mmpose.models import builder
from mmpose.models.builder import POSENETS
from mmpose.models.detectors.base import BasePose
import torch
# import smplx
from torch.nn import functional as F
from mmpose.models.egocentric.iknet.geometry import rotation_matrix_to_angle_axis

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

def rot6d_to_axis_angle(x):
    batch_size = x.shape[0]

    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix

    rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1)).cuda().float()], 2)  # 3x4 rotation matrix
    axis_angle = rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

@POSENETS.register_module()
class SimpleIkNet(nn.Module):
    def __init__(self, feat_dim=1024, joint_num=15, center_pose=True, smplx_config=None):
        """

        Args:
            feat_dim: input feature dimension
            joint_num: input 3d joint number
            smplx_config: smplx model setting, dict containing necessary information for initializing smplx model
        """

        super(SimpleIkNet, self).__init__()

        self.center_pose = center_pose

        if smplx_config is None:
            # use default setting
            smplx_config = dict(
                model_path='/CT/EgoMocap/work/smpl_models/models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.npz',
                model_type='smplx', use_pca=False, flat_hand_mean=True, num_betas=10
            )
        self.smplx = SMPLXLayer(**smplx_config)

        self.joint_num = joint_num
        self.body_feature = make_linear_layers([feat_dim, 512], relu_final=False, use_bn=False)
        self.root_pose_out = make_linear_layers([self.joint_num * 3 + 512, 6], relu_final=True, use_bn=False)
        self.body_pose_out = make_linear_layers(
            [512 + 3 * self.joint_num, 21 * 6], relu_final=False, use_bn=False)  # without root
        self.shape_out = make_linear_layers([feat_dim, smplx_config['num_betas']], relu_final=False,
                                            use_bn=False)
        self.global_transl_out = make_linear_layers([512 + 3 * self.joint_num, 3], relu_final=False, use_bn=False)
        self.feat_dim = feat_dim


    def forward(self, input):
        image_features = input['features']
        keypoints_3d = input['keypoints_pred']
        batch_size = image_features.shape[0]
        if self.center_pose:
            # put the pose to the center of image
            joint_center = (keypoints_3d[:, 7:8, :] + keypoints_3d[:, 11:12, :]) / 2
            keypoints_3d -= joint_center

        # shape parameter
        shape_param = self.shape_out(image_features)

        # body pose parameter
        body_pose_feature = self.body_feature(image_features)
        body_pose_feature = torch.cat((body_pose_feature, keypoints_3d.view(batch_size, -1)), -1)
        root_pose = self.root_pose_out(body_pose_feature)
        body_pose = self.body_pose_out(body_pose_feature)
        transl = self.global_transl_out(body_pose_feature)

        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(batch_size * 21, 6)).reshape(batch_size, -1)


        #if center pose, recover the real translation
        if self.center_pose:
            transl += joint_center[:, 0, :]

        # generate smplx info
        smpl_params = {
            'transl': transl,
            'body_pose': body_pose,
            'global_orient': root_pose,
            'betas': shape_param
        }
        smplx_info = self.smplx(**smpl_params)
        return_dict = {
            'body_pose': body_pose,
            'betas': shape_param,
            'transl': transl,
            'global_orient': root_pose,
            'keypoints_pred': smplx_info.joints,
            'vertices': smplx_info.vertices,
        }

        return return_dict