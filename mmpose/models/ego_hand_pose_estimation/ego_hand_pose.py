#  Copyright Jian Wang @ MPI-INF (c) 2023.

import warnings
from copy import deepcopy

import torch
import torch.nn as nn

from mmpose.models.builder import POSENETS, build_loss
from mmpose.models.ego_hand_pose_estimation.hands4whole_model import get_model
from mmpose.models.detectors.base import BasePose
try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class EgoHandPose(BasePose):
    def show_result(self, **kwargs):
        pass

    def __init__(self, pretrained=None,
                 loss_keypoint=dict(type='MPJPELoss', use_target_weight=False),
                 ):
        super(EgoHandPose, self).__init__()

        self.model = get_model('test')
        self.pretrained = pretrained

        if self.pretrained is not None:
            self.init_weights_original_model()
        else:
            print('init weights from default')

        self.loss = build_loss(loss_keypoint)

    def init_weights_original_model(self):
        """Weight initialization for model."""
        print('load pretrained model from {}'.format(self.pretrained))
        checkpoint = torch.load(self.pretrained)
        state_dict = checkpoint['network']
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}

        self.model.load_state_dict(new_state_dict, strict=False)

    @auto_fp16(apply_to=('left_hand_img', 'right_hand_img',))
    def forward(self,
                left_hand_img,
                right_hand_img,
                left_hand_transform=None,
                right_hand_transform=None,
                left_hand_keypoints_3d=None,
                right_hand_keypoints_3d=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        """Defines the computation performed at every call."""
        if return_loss:
            return self.forward_train(left_hand_img, right_hand_img, left_hand_transform, right_hand_transform,
                                      left_hand_keypoints_3d, right_hand_keypoints_3d, img_metas,
                                      **kwargs)
        return self.forward_test(
            left_hand_img, right_hand_img, left_hand_transform, right_hand_transform,
            img_metas, **kwargs)

    def forward_train(self, left_hand_image, right_hand_image, left_hand_transform, right_hand_transform,
                            left_hand_keypoints_3d, right_hand_keypoints_3d, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""

        left_hands_pred, right_hands_pred = self.run_model(left_hand_image, right_hand_image, left_hand_transform,
                                                           right_hand_transform, mode='train')

        left_hands_joints_pred = left_hands_pred['joint_cam_transformed']
        right_hands_joints_pred = right_hands_pred['joint_cam_transformed']

        # align left and right hands to root joint
        left_hands_joints_pred = left_hands_joints_pred - left_hands_joints_pred[:, 0:1, :]
        right_hands_joints_pred = right_hands_joints_pred - right_hands_joints_pred[:, 0:1, :]

        left_hands_joints_gt = left_hand_keypoints_3d - left_hand_keypoints_3d[:, 0:1, :]
        right_hands_joints_gt = right_hand_keypoints_3d - right_hand_keypoints_3d[:, 0:1, :]


        # calculate the loss
        losses = dict()
        losses['left_hand_loss'] = self.loss(left_hands_joints_pred, left_hands_joints_gt)
        losses['right_hand_loss'] = self.loss(right_hands_joints_pred, right_hands_joints_gt)

        return losses

    def forward_test(self, left_hand_image, right_hand_image, left_hand_transform, right_hand_transform, img_metas,
                     **kwargs):
        """Defines the computation performed at every call when testing."""
        left_hands_pred, right_hands_pred = self.run_model(left_hand_image, right_hand_image, left_hand_transform,
                                                           right_hand_transform)

        result = {'img_metas': img_metas}
        result['left_hands_preds'] = left_hands_pred
        result['right_hands_preds'] = right_hands_pred
        return result

    def run_model(self, left_hand_image, right_hand_image, left_hand_transform, right_hand_transform, mode='test'):
        assert left_hand_image.size(0) == right_hand_image.size(0)
        batch_size = left_hand_image.size(0)
        left_hand_image = torch.flip(left_hand_image, dims=[3])

        img = torch.cat((left_hand_image, right_hand_image), dim=0)
        inputs = {'img': img}
        targets = {}
        meta_info = {}
        hands_pred = self.model(inputs, targets, meta_info, mode='test')

        if mode == 'test':
            left_hands_pred = {key: value[:batch_size].detach().cpu() for key, value in hands_pred.items() if key != 'img'}
            right_hands_pred = {key: value[batch_size:].detach().cpu() for key, value in hands_pred.items() if key != 'img'}
        elif mode == 'train':
            left_hands_pred = {key: value[:batch_size] for key, value in hands_pred.items() if key != 'img'}
            right_hands_pred = {key: value[batch_size:] for key, value in hands_pred.items() if key != 'img'}
        else:
            raise Exception('mode should be either train or test')

        # transform the joints_cam
        # swap the left hand
        left_joint_cam_transformed = left_hands_pred['joint_cam'].clone()
        left_joint_cam_transformed[:, :, 0] *= -1
        left_joint_cam_transformed = self.transform_hand_pose(left_joint_cam_transformed, left_hand_transform)
        left_hands_pred['joint_cam_transformed'] = left_joint_cam_transformed
        right_joint_cam_transformed = self.transform_hand_pose(right_hands_pred['joint_cam'].clone(),
                                                               right_hand_transform)
        right_hands_pred['joint_cam_transformed'] = right_joint_cam_transformed
        return left_hands_pred, right_hands_pred

    def transform_hand_pose(self, hand_pose, transform):
        # hand pose shape: (batch_size, 21, 3)
        # transform shape: (batch_size, 4, 4)
        # transform hand pose to camera space
        transform = torch.asarray(transform).float().to(hand_pose.device)
        transformed_hand_pose = torch.bmm(transform[:, :3, :3], hand_pose.permute(0, 2, 1)).permute(0, 2, 1)
        # the translation part of the transformer is not used
        # transformed_hand_pose = transformed_hand_pose + transform[:, :3, 3:]
        return transformed_hand_pose
