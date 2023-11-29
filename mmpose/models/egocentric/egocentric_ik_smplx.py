#  Copyright Jian Wang @ MPI-INF (c) 2023.
import warnings

import mmcv
import numpy as np
import torch
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core import imshow_bboxes, imshow_keypoints
from .. import builder
from ..builder import POSENETS
from mmpose.models.detectors.base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class EgocentricIkSmplx(BasePose):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def show_result(self, **kwargs):
        pass

    def __init__(self,
                 pose_network=None,
                 pose_network_load_path=None,
                 freeze_pose_network=False,
                 ik_network=None,
                 smplx_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 ):
        super().__init__()
        self.fp16_enabled = False

        self.pose_network = builder.build_backbone(pose_network)
        self.freeze_pose_network = freeze_pose_network
        if self.freeze_pose_network:
            self.pose_network.eval()
            self.pose_network.requires_grad_(False)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.ik_network = builder.build_backbone(ik_network)
        self.smplx_loss = builder.build_loss(smplx_loss)
        self.pretrained = pose_network_load_path
        self.init_weights(self.pretrained)

    def init_weights(self, pretrained):
        """Weight initialization for model."""
        if pretrained is not None:
            state_dict = torch.load(pretrained)['state_dict']
            print('Loading pretrained model from {}'.format(pretrained))
            self.pose_network.load_state_dict(state_dict)

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                img_metas=None,
                return_loss=True,
                **kwargs):

        if return_loss:
            return self.forward_train(img, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, **kwargs)

    def forward_train(self, img, img_metas, **kwargs):
        target = kwargs
        """Defines the computation performed at every call when training."""
        if self.freeze_pose_network:
            self.pose_network.eval()
            self.pose_network.requires_grad_(False)
        output = self.pose_network.forward_test_with_features(img, img_metas, return_features=True, **kwargs)
        # use ik network to predict smplx parameters
        iknet_output = self.ik_network(output)
        # re-organize the iknet output and compute smplx loss
        smplx_loss = self.smplx_loss(iknet_output, target)
        return smplx_loss


    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        output = self.pose_network.forward_test_with_features(img, img_metas, return_features=True, **kwargs)
        # use ik network to predict smplx parameters
        iknet_output = self.ik_network(output)
        iknet_output['img_metas'] = img_metas
        return iknet_output