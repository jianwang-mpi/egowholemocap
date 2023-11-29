#  Copyright Jian Wang @ MPI-INF (c) 2023.
import warnings

import mmcv
import numpy as np
import torch
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow
from mmpose.models.builder import build_loss
from mmpose.core import imshow_bboxes, imshow_keypoints
from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.models import builder
from mmpose.models.builder import POSENETS
from mmpose.models.detectors.base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class RefineNet(BasePose):
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

    def __init__(self,
                 pose_backbone=None,
                 refinenet=None,
                 train_cfg=None,
                 test_cfg=None,
                 posenet_pretrained=None,
                 refinenet_pretrained=None,
                 freeze_backbone=False,
                 loss_keypoint=dict(type='MPJPELoss', use_target_weight=True),
                 ):
        super().__init__()
        self.fp16_enabled = False

        if pose_backbone is not None:
            self.pose_backbone = builder.build_backbone(pose_backbone)
        else:
            self.pose_backbone = None
        self.refinenet = builder.build_backbone(refinenet)
        self.freeze_backbone = freeze_backbone

        self.loss = build_loss(loss_keypoint)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.refinenet_pretrained = refinenet_pretrained
        self.posenet_pretrained = posenet_pretrained
        self.init_weights()

    def init_weights(self):
        """Weight initialization for model."""
        if self.refinenet_pretrained is None:
            self.refinenet.init_weights()
        else:
            state_dict = torch.load(self.pretrained)['state_dict']
            self.refinenet.load_state_dict(state_dict, strict=False)

        if self.posenet_pretrained is not None:
            state_dict = torch.load(self.posenet_pretrained)['state_dict']
            self.pose_backbone.load_state_dict(state_dict, strict=False)

    @auto_fp16(apply_to=('img', 'original_image', ))
    def forward(self,
                img,
                original_image,
                keypoints_3d=None,
                keypoints_3d_visible=None,
                img_metas=None,
                return_loss=True,
                return_features=False,
                **kwargs):
        if return_loss:
            return self.forward_train(img, original_image, keypoints_3d, keypoints_3d_visible, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, original_image, keypoints_3d, keypoints_3d_visible,
            img_metas, return_features=return_features, **kwargs)

    def forward_train(self, img, original_image, keypoints_3d, keypoints_3d_visible, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        if self.freeze_backbone:
            self.pose_backbone.requires_grad = False
        if self.pose_backbone is not None:
            x = self.pose_backbone(img)
            net_input = {
                'keypoints_3d': x,
                'original_image': original_image
            }
            output = self.refinenet(net_input)
        else:
            net_input = {
                'keypoints_3d': keypoints_3d,
                'original_image': original_image
            }
            output = self.refinenet(net_input)

        # if return loss
        losses = dict()

        keypoint_losses = self.get_loss(
            output, keypoints_3d, keypoints_3d_visible)
        losses.update(keypoint_losses)
        keypoint_accuracy = self.get_accuracy(
            output, keypoints_3d, keypoints_3d_visible)
        losses.update(keypoint_accuracy)

        return losses

    def get_loss(self, output, keypoints_3d_gt, keypoint_3d_visible):
        losses = dict()

        assert keypoints_3d_gt.dim() == 3
        losses['mpjpe_loss'] = self.loss(output['keypoints_pred'], keypoints_3d_gt, keypoint_3d_visible[:, :, None])

        return losses

    def get_accuracy(self, output, keypoints_3d_gt, keypoint_3d_visible):
        accuracy = dict()
        keypoints_pred = output['keypoints_pred']

        mpjpe = keypoint_mpjpe(
            keypoints_pred.detach().cpu().numpy(),
            keypoints_3d_gt.detach().cpu().numpy(),
            mask=keypoint_3d_visible.detach().cpu().numpy().astype(np.bool), alignment='none')
        accuracy['mpjpe'] = float(mpjpe)

        keypoints_3d_w_noise = output['keypoints_3d_w_noise']
        mpjpe_w_noise = keypoint_mpjpe(
            keypoints_3d_w_noise.detach().cpu().numpy(),
            keypoints_3d_gt.detach().cpu().numpy(),
            mask=keypoint_3d_visible.detach().cpu().numpy().astype(np.bool), alignment='none')
        accuracy['mpjpe_w_noise'] = float(mpjpe_w_noise)

        return accuracy

    def forward_test(self, img, original_image, keypoints_3d, keypoints_3d_visible, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape

        result = {'img_metas': img_metas}

        if self.pose_backbone is not None:
            x = self.pose_backbone(img)
            net_input = {
                'keypoints_3d': x,
                'original_image': original_image
            }
            output = self.refinenet(net_input)
        else:
            net_input = {
                'keypoints_3d': keypoints_3d,
                'original_image': original_image
            }
            output = self.refinenet(net_input)
        result.update(output)
        return result


    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):


        return None
