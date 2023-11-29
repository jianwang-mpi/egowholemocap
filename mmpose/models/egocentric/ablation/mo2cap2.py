#  Copyright Jian Wang @ MPI-INF (c) 2023.
import warnings

import torch.nn as nn
from mmcv.runner.checkpoint import load_checkpoint

from mmpose.core.evaluation.top_down_eval import _get_softargmax_preds
from mmpose.models.builder import HEADS, build_posenet, build_loss, POSENETS
from mmpose.models.detectors.base import BasePose
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated

# from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
#                       constant_init, normal_init)

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, expansion=2, downsample=None):
        super(Bottleneck, self).__init__()
        BN_MOMENTUM = 0.1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FinalLayer(nn.Module):
    def __init__(self, input_features=256, linear_in_features=16 * 64):
        super(FinalLayer, self).__init__()
        self.final_conv1 = nn.Conv2d(in_channels=input_features, out_channels=128, kernel_size=4, stride=2, padding=1,
                                     bias=False)
        self.final_bn1 = nn.BatchNorm2d(num_features=128)
        self.final_relu1 = nn.ReLU()
        self.final_conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True)
        self.final_relu2 = nn.ReLU()
        self.final_depth = nn.Linear(in_features=linear_in_features, out_features=15, bias=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.final_conv1(x)
        x = self.final_bn1(x)
        x = self.final_relu1(x)
        x = self.final_conv2(x)
        x = self.final_relu2(x)
        x = x.view(size=(batch_size, -1))
        x = self.final_depth(x)
        return x


@HEADS.register_module()
class DistanceEstimator(nn.Module):
    def __init__(self, input_feature_dim=768, linear_in_features=16 * 64):
        super(DistanceEstimator, self).__init__()
        BN_MOMENTUM = 0.1
        downsample1 = nn.Sequential(
            nn.Conv2d(input_feature_dim, 512 * 2,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512 * 2, momentum=BN_MOMENTUM),
        )
        self.resblock1 = Bottleneck(inplanes=input_feature_dim, planes=512, expansion=2, downsample=downsample1)
        downsample2 = nn.Sequential(
            nn.Conv2d(512 * 2, 256 * 1,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256 * 1, momentum=BN_MOMENTUM),
        )
        self.init_weights(self.resblock1)
        self.resblock2 = Bottleneck(inplanes=1024, planes=256, expansion=1, downsample=downsample2)
        self.init_weights(self.resblock2)
        self.final_layer = FinalLayer(input_features=256, linear_in_features=linear_in_features)
        self.init_weights(self.final_layer)

    def init_weights(self, network):
        for m in network.children():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.final_layer(x)
        return x


@POSENETS.register_module()
class Mo2Cap2(BasePose):
    def __init__(self,
                 pose_2d_module_cfg,
                 pose_2d_module_load_path=None,
                 joint_num=15,
                 input_feature_dim=768,
                 linear_in_features=16 * 64,
                 freeze_pose_2d_module=True,
                 distance_loss_cfg=None,
                 fisheye_camera_path=None,
                 train_cfg=None,
                 test_cfg=None
                 ):
        super(Mo2Cap2, self).__init__()
        self.pose_2d_module = build_posenet(pose_2d_module_cfg)
        if pose_2d_module_load_path is not None:
            load_checkpoint(self.pose_2d_module, pose_2d_module_load_path, map_location='cpu')
        self.distance_module = DistanceEstimator(input_feature_dim=input_feature_dim,
                                                 linear_in_features=linear_in_features)
        if distance_loss_cfg is None:
            distance_loss_cfg = dict(type='MSELoss', loss_weight=1.0, use_target_weight=False)
        self.distance_loss = build_loss(distance_loss_cfg)

        self.freeze_pose_2d_module = freeze_pose_2d_module
        # logger = logging.getLogger()
        self.fisheye_camera = FishEyeCameraCalibrated(fisheye_camera_path)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.joint_num = joint_num

    @auto_fp16(apply_to=('img',))
    def forward(self,
                img,
                keypoints_3d=None,
                keypoints_3d_visible=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        if return_loss:
            return self.forward_train(img, keypoints_3d, keypoints_3d_visible, img_metas, **kwargs)
        return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, keypoints_3d, keypoints_3d_visible, img_metas, **kwargs):
        """Here we only train the distance module"""
        if self.freeze_pose_2d_module:
            self.pose_2d_module.requires_grad = False
        else:
            self.pose_2d_module.requires_grad = True
        features = self.pose_2d_module.backbone(img)
        distance = self.distance_module(features)
        # get distance gt from keypoints 3d with fisheye camera model
        batch_size, joint_num, _ = keypoints_3d.shape
        assert joint_num == self.joint_num
        assert _ == 3
        keypoints_3d_gt = keypoints_3d.reshape((-1, 3))
        keypoints_2d_gt, distance_gt = self.fisheye_camera.world2camera_pytorch_with_depth(keypoints_3d_gt)
        distance_gt = distance_gt.reshape((-1, self.joint_num)).float()
        # if return loss
        losses = dict()
        losses['distance_loss'] = self.distance_loss(distance, distance_gt)

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape

        result = {'img_metas': img_metas}
        pred_hm = self.pose_2d_module.forward_with_grad(img)
        features = self.pose_2d_module.backbone(img)
        distance = self.distance_module(features)
        pred_hm = pred_hm.detach().cpu().numpy()
        distance = distance.detach().cpu().numpy()
        # get pred keypoints from heatmap and distance

        keypoints_2d, _ = _get_softargmax_preds(pred_hm)
        B, K, H, W = pred_hm.shape
        assert H == 64
        keypoints_2d = keypoints_2d * (1024 / H)
        keypoints_2d[:, :, 0] += 128
        keypoints_2d = keypoints_2d.reshape((-1, 2))
        distance = distance.reshape((B * self.joint_num))
        keypoints_3d = self.fisheye_camera.camera2world(keypoints_2d, distance)
        keypoints_3d = keypoints_3d.reshape((B, self.joint_num, 3))
        result['keypoints_pred'] = keypoints_3d
        return result

    def init_weights(self):
        """Initialize model weights."""
        pass

    def show_result(self, **kwargs):
        pass
