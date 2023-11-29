#  Copyright Jian Wang @ MPI-INF (c) 2023.

import torch.nn as nn

from mmpose.core import imshow_bboxes, imshow_keypoints
from mmpose.models import builder
from mmpose.models.builder import POSENETS
from mmpose.models.detectors.base import BasePose
import torch
# import smplx
from torch.nn import functional as F
from mmpose.models.egocentric.iknet.geometry import rotation_matrix_to_angle_axis


def make_linear_layers(feat_dims, activation_type='relu', final_activation=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))
        if use_bn:
            layers.append(nn.BatchNorm1d(feat_dims[i + 1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims) - 2 and final_activation):
            if activation_type == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation_type == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise NotImplementedError

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