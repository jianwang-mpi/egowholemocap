#  Copyright Jian Wang @ MPI-INF (c) 2023.

# regress the 3d heatmap under the fisheye camera view and give 3d pose prediction
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_upsample_layer,
                      constant_init, normal_init)
from scipy.ndimage import gaussian_filter

from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.models.builder import HEADS, build_loss
from mmpose.models.utils.ops import resize
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated


def soft_argmax_3d(heatmap3d):
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth * height * width))
    heatmap3d = F.softmax(heatmap3d, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

    accu_x = heatmap3d.sum(dim=(2, 3))
    accu_y = heatmap3d.sum(dim=(2, 4))
    accu_z = heatmap3d.sum(dim=(3, 4))

    accu_x = accu_x * torch.arange(width).float().cuda()[None, None, :]
    accu_y = accu_y * torch.arange(height).float().cuda()[None, None, :]
    accu_z = accu_z * torch.arange(depth).float().cuda()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out


@HEADS.register_module()
class Heatmap3DNet(nn.Module):
    def __init__(self,
                 in_channels=768,
                 num_deconv_layers=2,
                 num_deconv_filters=(1024, 15 * 64),
                 num_deconv_kernels=(4, 4),
                 out_channels=15 * 64,
                 heatmap_shape=(64, 64, 64),
                 voxel_size=(2, 2, 2),
                 fisheye_model_path=None,
                 joint_num=15,
                 loss_keypoint=dict(type='MPJPELoss', use_target_weight=True),
                 input_transform=None,
                 in_index=None,
                 train_cfg=None,
                 test_cfg=None
                 ):
        super(Heatmap3DNet, self).__init__()
        self.in_channels = in_channels
        self.deconv = self._make_deconv_layer(num_deconv_layers, num_deconv_filters, num_deconv_kernels)
        self.final_conv = nn.Conv2d(num_deconv_filters[-1], out_channels, kernel_size=1, stride=1, padding=0)
        self.heatmap_shape = heatmap_shape
        self.fisheye_model = FishEyeCameraCalibrated(fisheye_model_path)
        self.joint_num = joint_num
        self.voxel_size = voxel_size
        self.loss = build_loss(loss_keypoint)

        self.input_transform = input_transform
        self.in_index = in_index
        logger = logging.getLogger()
        logger.info(f'Input transform: {self.input_transform}')
        logger.info(f'Input index: {self.in_index}')

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list) or self.input_transform is None:
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=False) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
            # suppose input size is 16 * 16
            B, C, H, W = inputs.shape
            if H != 16 or W != 16:
                # print('The input size is not 16 * 16, resize it to 16 * 16')
                inputs = F.interpolate(inputs, size=(16, 16), mode='bilinear', align_corners=False)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, img_feat):
        img_feat = self._transform_inputs(img_feat)
        joint_3d_heatmap = self.deconv(img_feat).view(-1, self.joint_num, self.heatmap_shape[0],
                                                      self.heatmap_shape[1], self.heatmap_shape[2])
        joint_coord = soft_argmax_3d(joint_3d_heatmap)

        # the joint coord is under 64 * 64 * 64 space, we need to convert it to real world space
        resize_z = self.voxel_size[2] / self.heatmap_shape[2]
        # resize the depth to real world space
        joint_coord[:, :, 2] = joint_coord[:, :, 2] * resize_z
        # resize x and y to image space
        joint_coord[:, :, 0] = joint_coord[:, :, 0] / self.heatmap_shape[0] * 1024 + 128
        joint_coord[:, :, 1] = joint_coord[:, :, 1] / self.heatmap_shape[1] * 1024

        # convert joint coord in fisheye space to joint coord in camera space
        joint_coord = self.fisheye2camera(joint_coord)

        return joint_coord

    def fisheye2camera(self, joint_coord):
        # joint_coord: [batch_size, joint_num, 3]
        # joint_coord_cam: [batch_size, joint_num, 3]
        batch_size = joint_coord.shape[0]
        joint_coord_xy = joint_coord[:, :, :2].view(batch_size * self.joint_num, 2)
        joint_coord_z = joint_coord[:, :, 2].view(batch_size * self.joint_num)
        joint_coord_cam = self.fisheye_model.camera2world_pytorch(joint_coord_xy, joint_coord_z)
        joint_coord_cam = joint_coord_cam.view(batch_size, self.joint_num, 3)
        joint_coord_cam = joint_coord_cam.contiguous()
        return joint_coord_cam

    def get_loss(self, output, keypoints_3d, keypoint_3d_visible=None, **kwargs):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_keypoint_pos: 3

        Args:
            output (torch.Tensor[N,K,3): Output keypoints.
            keypoints_3d (torch.Tensor[N,K,3]): Target keypoints.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert keypoints_3d.dim() == 3
        losses['mpjpe_loss'] = self.loss(output, keypoints_3d, keypoint_3d_visible[:, :, None])

        return losses

    def get_accuracy(self, output, keypoints_3d, keypoint_3d_visible=None):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        """

        accuracy = dict()
        N, K, _ = output.shape

        mpjpe = keypoint_mpjpe(
            output.detach().cpu().numpy(),
            keypoints_3d.detach().cpu().numpy(),
            mask=keypoint_3d_visible.detach().cpu().numpy().astype(np.bool), alignment='none')
        accuracy['mpjpe'] = float(mpjpe)

        return accuracy

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        img_feat = self._transform_inputs(x)
        joint_3d_heatmap = self.deconv(img_feat).view(-1, self.joint_num, self.heatmap_shape[0],
                                                      self.heatmap_shape[1], self.heatmap_shape[2])
        joint_coord = soft_argmax_3d(joint_3d_heatmap)

        joint_coord_in_voxel = joint_coord.detach().clone()

        # the joint coord is under 64 * 64 * 64 space, we need to convert it to real world space
        resize_z = self.voxel_size[2] / self.heatmap_shape[2]
        # resize the depth to real world space
        joint_coord[:, :, 2] = joint_coord[:, :, 2] * resize_z
        # resize x and y to image space
        joint_coord[:, :, 0] = joint_coord[:, :, 0] / self.heatmap_shape[0] * 1024 + 128
        joint_coord[:, :, 1] = joint_coord[:, :, 1] / self.heatmap_shape[1] * 1024

        # convert joint coord in fisheye space to joint coord in camera space
        joint_coord = self.fisheye2camera(joint_coord)
        result = {'keypoints_pred': joint_coord.detach().cpu().numpy()}
        if 'return_heatmap' in self.test_cfg and self.test_cfg['return_heatmap'] is True:
            result['heatmap'] = joint_3d_heatmap.detach().cpu().numpy()

        if 'return_confidence' in self.test_cfg and self.test_cfg['return_confidence'] is True:
            # calculate confidence for each joint
            # joint_voxel shape: (batch_size, joint_number, 3)
            # heatmap shape: (batch_size, joint_number, Depth, Height, Width)
            joint_3d_heatmap_confidence = joint_3d_heatmap.detach()
            joint_voxel = joint_coord_in_voxel.detach()
            batch_size, joint_num, depth, height, width = joint_3d_heatmap_confidence.shape
            # use grid_sample to get the confidence
            # joint_voxel shape: (batch_size, joint_number, 3)
            # add gaussian filter
            joint_3d_heatmap_confidence = joint_3d_heatmap_confidence.cpu().numpy()
            joint_3d_heatmap_confidence = gaussian_filter(joint_3d_heatmap_confidence, sigma=self.test_cfg['sigma'],
                                                          axes=(2, 3, 4))
            joint_3d_heatmap_confidence = torch.from_numpy(joint_3d_heatmap_confidence).to(joint_voxel.device)

            joint_3d_heatmap_confidence = joint_3d_heatmap_confidence.view(batch_size * joint_num, 1, depth, height,
                                                                           width)

            joint_voxel = joint_voxel.view(batch_size * joint_num, 1, 1, 1, 3)
            assert self.heatmap_shape[0] == self.heatmap_shape[1] == self.heatmap_shape[2]
            joint_voxel = joint_voxel / self.heatmap_shape[2] * 2 - 1
            joint_3d_heatmap_confidence = torch.nn.functional.grid_sample(
                joint_3d_heatmap_confidence, joint_voxel, align_corners=False)
            joint_3d_heatmap_confidence = joint_3d_heatmap_confidence.view(batch_size, joint_num)
            result['keypoint_confidence'] = joint_3d_heatmap_confidence.detach().cpu().numpy()
            print(result['keypoint_confidence'])
        if 'return_2d_heatmap' in self.test_cfg and self.test_cfg['return_2d_heatmap'] is True:
            # calculate confidence for each joint
            # joint_voxel shape: (batch_size, joint_number, 3)
            # heatmap shape: (batch_size, joint_number, Depth, Height, Width)
            joint_3d_heatmap_confidence = joint_3d_heatmap.detach()
            joint_3d_heatmap_confidence = joint_3d_heatmap_confidence.cpu().numpy()
            joint_3d_heatmap_confidence = gaussian_filter(joint_3d_heatmap_confidence, sigma=self.test_cfg['sigma'],
                                                          axes=(2, 3, 4))
            # convert 3d heatmap to 2d heatmap alone the z axis
            joint_2d_heatmap_confidence = joint_3d_heatmap_confidence.sum(axis=2)
            result['heatmap_2d'] = joint_2d_heatmap_confidence
        return result

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        normal_init(self.final_conv, std=0.001, bias=0)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding
