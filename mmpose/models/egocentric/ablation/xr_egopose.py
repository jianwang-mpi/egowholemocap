#  Copyright Jian Wang @ MPI-INF (c) 2023.
#  Copyright Jian Wang @ MPI-INF (c) 2023.

# regress the 3d heatmap under the fisheye camera view and give 3d pose prediction
import warnings
from collections import OrderedDict
from mmcv.runner.checkpoint import load_checkpoint
import numpy as np
import torch
import torch.nn as nn
# from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
#                       constant_init, normal_init)
from torch.nn.utils import weight_norm

from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.models.builder import HEADS, build_posenet, build_loss, POSENETS
from mmpose.models.detectors.base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


class AutoEncoder(nn.Module):

    def make_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        torch.nn.init.xavier_normal_(conv.weight)
        conv = weight_norm(conv)
        bn = torch.nn.BatchNorm2d(num_features=out_channels)
        relu = torch.nn.LeakyReLU(negative_slope=0.2)
        if self.with_bn:
            return torch.nn.Sequential(conv, bn, relu)
        else:
            return torch.nn.Sequential(conv, relu)

    def make_deconv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        conv = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding)
        torch.nn.init.xavier_normal_(conv.weight)
        conv = weight_norm(conv)
        bn = torch.nn.BatchNorm2d(num_features=out_channels)
        relu = torch.nn.LeakyReLU(negative_slope=0.2)
        if self.with_bn:
            return torch.nn.Sequential(conv, bn, relu)
        else:
            return torch.nn.Sequential(conv, relu)

    def make_fc_layer(self, in_feature, out_feature, with_relu=True):
        modules = OrderedDict()
        fc = torch.nn.Linear(in_feature, out_feature)
        torch.nn.init.xavier_normal_(fc.weight)
        fc = weight_norm(fc)
        modules['fc'] = fc
        bn = torch.nn.BatchNorm1d(num_features=out_feature)
        relu = torch.nn.LeakyReLU(negative_slope=0.2)
        if self.with_bn is True:
            modules['bn'] = bn
        else:
            print('no bn')
        if with_relu is True:
            modules['relu'] = relu
        else:
            print('no pose relu')
        return torch.nn.Sequential(modules)

    def __init__(self, hidden_size=20, with_bn=True, with_pose_relu=True):
        super(AutoEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.with_bn = with_bn
        self.with_pose_relu = with_pose_relu
        self.conv1 = self.make_conv_layer(in_channels=15, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = self.make_conv_layer(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = self.make_conv_layer(in_channels=128, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.fc1 = self.make_fc_layer(in_feature=32768, out_feature=2048)
        self.fc2 = self.make_fc_layer(in_feature=2048, out_feature=512)
        self.fc3 = self.make_fc_layer(in_feature=512, out_feature=self.hidden_size)

        ## pose decoder
        self.pose_fc1 = self.make_fc_layer(self.hidden_size, 32, with_relu=self.with_pose_relu)
        self.pose_fc2 = self.make_fc_layer(32, 32, with_relu=self.with_pose_relu)
        # self.pose_fc3 = self.make_fc_layer(32, 45)
        self.pose_fc3 = torch.nn.Linear(32, 45)
        torch.nn.init.xavier_normal_(self.pose_fc3.weight)
        self.pose_fc3 = weight_norm(self.pose_fc3)

        # heatmap decoder
        self.heatmap_fc1 = self.make_fc_layer(self.hidden_size, 512)
        self.heatmap_fc2 = self.make_fc_layer(512, 2048)
        self.heatmap_fc3 = self.make_fc_layer(2048, 32768)

        self.deconv1 = self.make_deconv_layer(512, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = self.make_deconv_layer(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = self.make_deconv_layer(64, 15, kernel_size=4, stride=2, padding=1)

    def predict_pose(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        z = self.fc3(x)
        # pose decode
        x_pose = self.pose_fc1(z)
        x_pose = self.pose_fc2(x_pose)
        x_pose = self.pose_fc3(x_pose)
        return x_pose.view(-1, 15, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        z = self.fc3(x)
        # pose decode
        x_pose = self.pose_fc1(z)
        x_pose = self.pose_fc2(x_pose)
        x_pose = self.pose_fc3(x_pose)

        # heatmap decode

        x_hm = self.heatmap_fc1(z)
        x_hm = self.heatmap_fc2(x_hm)
        x_hm = self.heatmap_fc3(x_hm)
        x_hm = x_hm.view(-1, 512, 8, 8)
        x_hm = self.deconv1(x_hm)
        x_hm = self.deconv2(x_hm)
        x_hm = self.deconv3(x_hm)

        return x_pose.view(-1, 15, 3), x_hm


@POSENETS.register_module()
class XREgoPose(BasePose):


    def __init__(self,
                 pose_2d_module_cfg,
                 pose_2d_module_load_path=None,
                 joint_num=15,
                 w_hm_pred=1,
                 w_hm_recon=1,
                 w_joints=1,
                 freeze_pose_2d_module=False,
                 heatmap_pred_loss_cfg=None,
                 heatmap_recon_loss_cfg=None,
                 joints_3d_loss_cfg=None,
                 train_cfg=None,
                 test_cfg=None
                 ):
        super(XREgoPose, self).__init__()
        self.pose_2d_module = build_posenet(pose_2d_module_cfg)
        if pose_2d_module_load_path is not None:
            load_checkpoint(self.pose_2d_module, pose_2d_module_load_path, map_location='cpu')
        self.auto_encoder = AutoEncoder()
        if heatmap_recon_loss_cfg is None:
            heatmap_recon_loss_cfg = dict(type='JointsMSELoss', use_target_weight=True)
        if heatmap_pred_loss_cfg is None:
            heatmap_pred_loss_cfg = dict(type='JointsMSELoss', use_target_weight=True)
        if joints_3d_loss_cfg is None:
            joints_3d_loss_cfg = dict(type='MPJPELoss', use_target_weight=True)
        self.mpjpe_loss = build_loss(joints_3d_loss_cfg)
        self.heatmap_recon_loss = build_loss(heatmap_recon_loss_cfg)
        self.heatmap_pred_loss = build_loss(heatmap_pred_loss_cfg)

        self.freeze_pose_2d_module = freeze_pose_2d_module
        # logger = logging.getLogger()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.joint_num = joint_num
        self.w_hm_pred = w_hm_pred
        self.w_hm_recon = w_hm_recon
        self.w_joints = w_joints

    @auto_fp16(apply_to=('img',))
    def forward(self,
                img,
                keypoints_3d=None,
                keypoints_3d_visible=None,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_features=False,
                **kwargs):
        if return_loss:
            return self.forward_train(img, keypoints_3d, keypoints_3d_visible, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_features=return_features, **kwargs)

    def forward_train(self, img, keypoints_3d, keypoints_3d_visible,
                      target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        if self.freeze_pose_2d_module:
            self.pose_2d_module.requires_grad = False
        else:
            self.pose_2d_module.requires_grad = True
        pred_hm = self.pose_2d_module.forward_with_grad(img)
        pose, recon_hm = self.auto_encoder(pred_hm)
        # if return loss
        losses = dict()
        losses['pred_hm_loss'] = self.heatmap_pred_loss(pred_hm, target, target_weight) * self.w_hm_pred
        losses['recon_hm_loss'] = self.heatmap_recon_loss(recon_hm, pred_hm, target_weight) * self.w_hm_recon
        losses['joint_3d_loss'] = self.mpjpe_loss(pose, keypoints_3d, keypoints_3d_visible[:, :, None]) * self.w_joints

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape

        result = {'img_metas': img_metas}
        pred_hm = self.pose_2d_module.forward_with_grad(img)
        pose, recon_hm = self.auto_encoder(pred_hm)
        pose = pose.detach().cpu().numpy()
        result['keypoints_pred'] = pose
        return result


    def init_weights(self):
        """Initialize model weights."""
        pass
    def show_result(self, **kwargs):
        pass