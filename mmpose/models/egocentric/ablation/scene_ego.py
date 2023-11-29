#  Copyright Jian Wang @ MPI-INF (c) 2023.
#  Copyright Jian Wang @ MPI-INF (c) 2023.

# regress the 3d heatmap under the fisheye camera view and give 3d pose prediction
import warnings
from collections import OrderedDict
from mmcv.runner.checkpoint import load_checkpoint
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.models.builder import HEADS, build_posenet, build_loss, POSENETS, build_backbone
from mmpose.models.detectors.base import BasePose
from mmpose.models.egocentric.ablation.voxel_net_depth import VoxelNetwork

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class SceneEgo(BasePose):
    def __init__(self,
                 pose_2d_module_cfg,
                 pose_2d_module_load_path=None,
                 joint_num=15,
                 w_joints=1,
                 freeze_backbone=False,
                 joints_3d_loss_cfg=None,
                 batch_size=64,
                 camera_calibration_file_path=None,
                 train_cfg=None,
                 test_cfg=None
                 ):
        super(SceneEgo, self).__init__()
        self.pose_2d_module = build_posenet(pose_2d_module_cfg)
        if pose_2d_module_load_path is not None:
            load_checkpoint(self.pose_2d_module, pose_2d_module_load_path, map_location='cpu')
        # extract the backbone network from the pose_2d_module
        self.backbone = self.pose_2d_module.backbone

        if joints_3d_loss_cfg is None:
            joints_3d_loss_cfg = dict(type='MPJPELoss', use_target_weight=True)
        self.mpjpe_loss = build_loss(joints_3d_loss_cfg)

        self.freeze_backbone = freeze_backbone
        # logger = logging.getLogger()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.joint_num = joint_num

        self.w_joints = w_joints

        # init voxel network
        self.voxel_network = VoxelNetwork(batch_size=batch_size,
                                          camera_calibration_file_path=camera_calibration_file_path)

    @auto_fp16(apply_to=('img',))
    def forward(self,
                img,
                keypoints_3d=None,
                keypoints_3d_visible=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        if return_loss:
            return self.forward_train(img, keypoints_3d, keypoints_3d_visible, img_metas,
                                      **kwargs)
        return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, keypoints_3d, keypoints_3d_visible, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        if self.freeze_backbone:
            self.backbone.requires_grad = False
        else:
            self.backbone.requires_grad = True
        features = self.backbone(img)
        pose = self.voxel_network(features)
        # if return loss
        losses = dict()
        losses['joint_3d_loss'] = self.mpjpe_loss(pose, keypoints_3d, keypoints_3d_visible[:, :, None]) * self.w_joints

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape

        result = {'img_metas': img_metas}
        features = self.backbone(img)
        pose = self.voxel_network(features)
        pose = pose.detach().cpu().numpy()
        result['keypoints_pred'] = pose
        return result


    def init_weights(self):
        """Initialize model weights."""
        pass
    def show_result(self, **kwargs):
        pass