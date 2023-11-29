#  Copyright Jian Wang @ MPI-INF (c) 2023.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)

from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
from mmpose.models.heads.topdown_heatmap_base_head import TopdownHeatmapBaseHead


@HEADS.register_module()
class Regress3DPoseSimpleHead(TopdownHeatmapBaseHead):
    """Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_joint_num,
                 num_conv_layers=3,
                 num_conv_filters=(256, 256, 256),
                 num_conv_kernels=(3, 3, 3),
                 num_conv_padding=(1, 1, 1),
                 num_conv_stride=(2, 2, 2),
                 num_fc_layers=2,
                 num_fc_features=(262144, 1024, 45),
                 extra=None,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 with_bn=True,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.with_bn = with_bn
        self.in_channels = in_channels
        self.out_joint_num = out_joint_num
        assert out_joint_num*3 == num_fc_features[-1]

        self.loss = build_loss(loss_keypoint)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_conv_layers > 0:
            self.conv_layers = self._make_conv_layer(
                num_conv_layers,
                num_conv_filters,
                num_conv_kernels,
                num_conv_padding,
                num_conv_stride,
                with_bn=with_bn
            )
        elif num_conv_layers == 0:
            self.conv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_conv_layers ({num_conv_layers}) should >= 0.')

        self.fc_layers = self._make_fc_layer(num_fc_layers, num_fc_features, regression_final=True, with_bn=with_bn)

    def get_loss(self, output, keypoints_3d, target_weight=None):
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
        losses['mpjpe_loss'] = self.loss(output, keypoints_3d, target_weight)

        return losses

    def get_accuracy(self, output, keypoints_3d, target_weight):
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
            mask=np.ones((N, K)).astype(np.bool), alignment='none')
        accuracy['mpjpe'] = float(mpjpe)

        return accuracy

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        B = x.shape[0]
        x = self.conv_layers(x)
        x = x.view(B, -1)
        x = self.fc_layers(x)
        x = x.view(B, self.out_joint_num, 3)
        return x

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        output_joints = output.detach().cpu().numpy()
        return output_joints

    def forward_with_feature(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        B = x.shape[0]
        x = self.conv_layers(x)
        x = x.view(B, -1)
        features = x.clone()
        x = self.fc_layers(x)
        x = x.view(B, self.out_joint_num, 3)
        return x, features


    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    @staticmethod
    def _get_conv_cfg(conv_kernel):
        """Get configurations for deconv layers."""
        if conv_kernel == 4:
            padding = 1
        elif conv_kernel == 3:
            padding = 1
        elif conv_kernel == 2:
            padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({conv_kernel}).')
        return conv_kernel, padding

    def _make_conv_layer(self, num_layers, num_filters, num_kernels, num_padding, num_stride, with_bn=True):
        """Make conv layers."""
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
            planes = num_filters[i]
            kernel_size = num_kernels[i]
            padding_size = num_padding[i]
            stride_size = num_stride[i]
            layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel_size,
                    stride=stride_size,
                    padding=padding_size,
                    bias=False))
            if with_bn:
                layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def _make_fc_layer(self, num_fc_layers, num_fc_features, regression_final=True, with_bn=True):
        """Make fc layer"""
        if num_fc_layers != len(num_fc_features) - 1:
            error_msg = f'num_layers({num_fc_layers}) ' \
                        f'!= length of num_filters({len(num_fc_features)}) - 1'
            raise ValueError(error_msg)
        layers = []
        for i in range(num_fc_layers):
            layers.append(
                nn.Linear(in_features=num_fc_features[i], out_features=num_fc_features[i+1]))
            if with_bn:
                layers.append(nn.BatchNorm1d(num_fc_features[i+1]))
            layers.append(nn.ReLU(inplace=True))
        if regression_final is True:
            layers.pop(-1)
        return nn.Sequential(*layers)


    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.conv_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.fc_layers.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
