#  Copyright Jian Wang @ MPI-INF (c) 2023.

import numpy as np
import smplx
import torch
import torch.nn as nn

from mmpose.data.keypoints_mapping.mo2cap2 import mo2cap2_parents
from mmpose.models.builder import POSENETS
from mmpose.models.egocentric.iknet.network_utils import make_linear_layers
from mmpose.models.egocentric.iknet.network_utils import rot6d_to_axis_angle


from mmpose.models.egocentric.iknet.smplx_layer import SMPLXLayer


def xyz_to_delta_batch(xyz, parents):
    """
    Convert joint coordinates into bone directions which are NOT normed.

    Parameters
    ----------
    xyz : [N, J, 3]
      Joint coordinates.
    parents : list
      Parent joints list.

    Returns
    -------
    [N, J, 3]
      Bone directions.
    """
    delta = []
    for c, p in parents:
        if p is None:
            delta.append(xyz[:, c])
        else:
            delta.append(xyz[:, c] - xyz[:, p])
    delta = torch.stack(delta, 1)
    return delta


@POSENETS.register_module()
class IkNetPose(nn.Module):
    def __init__(self, body_feature_network_layers=(15 * 7, 512, 512, 512, 512),
                 root_pose_network_layers=(512, 6),
                 body_pose_network_layers=(512, 21 * 6),
                 shape_network_layers=(512, 10),
                 transl_network_layers=(512, 3),
                 joint_num=15,
                 smplx_config=None):
        """

        Args:
            joint_num: input 3d joint number
            smplx_config: smplx model setting, dict containing necessary information for initializing smplx model
        """

        super(IkNetPose, self).__init__()

        if smplx_config is None:
            # use default setting
            smplx_config = dict(
                model_path='/CT/EgoMocap/work/smpl_models/models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.npz',
                model_type='smplx', use_pca=False, flat_hand_mean=True, num_betas=10
            )
        self.smplx = SMPLXLayer(**smplx_config)
        # self.smplx = smplx.create(**smplx_config)

        self.joint_num = joint_num
        self.body_feature = make_linear_layers(body_feature_network_layers, final_activation=True, use_bn=True,
                                               activation_type='sigmoid')
        self.root_pose_out = make_linear_layers(root_pose_network_layers, final_activation=False, use_bn=False,
                                                activation_type='sigmoid')
        self.body_pose_out = make_linear_layers(
            body_pose_network_layers, final_activation=False, use_bn=False, activation_type='sigmoid')  # without root
        self.shape_out = make_linear_layers(shape_network_layers, final_activation=False, use_bn=False,
                                            activation_type='sigmoid')
        self.global_transl_out = make_linear_layers(transl_network_layers, final_activation=False, use_bn=False,
                                                    activation_type='sigmoid')

    def forward(self, input):
        keypoints_3d = input['keypoints_pred']
        batch_size = keypoints_3d.shape[0]

        # generate network input
        delta = xyz_to_delta_batch(keypoints_3d, mo2cap2_parents)
        length = torch.linalg.norm(delta, dim=-1, keepdims=True)
        delta = delta / length

        network_input = torch.cat(
            [
                torch.reshape(keypoints_3d, [batch_size, self.joint_num * 3]),
                torch.reshape(delta, [batch_size, self.joint_num * 3]),
                torch.reshape(length, [batch_size, self.joint_num])
            ], dim=-1
        )



        # body pose parameter
        body_pose_feature = self.body_feature(network_input)

        root_pose = self.root_pose_out(body_pose_feature)
        body_pose = self.body_pose_out(body_pose_feature)
        transl = self.global_transl_out(body_pose_feature)

        # shape parameter
        shape_param = self.shape_out(body_pose_feature)

        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(batch_size * 21, 6)).reshape(batch_size, -1)

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
