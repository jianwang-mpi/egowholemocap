#  Copyright Jian Wang @ MPI-INF (c) 2023.

import warnings
import math
import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow
import torch.nn as nn
from mmpose.core import imshow_bboxes, imshow_keypoints
from .. import builder
from ..builder import POSENETS
from mmpose.models.detectors.base import BasePose
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated
from scipy.spatial.transform import Rotation as R
import open3d
import torch

@POSENETS.register_module()
class Fisheye2Sphere(nn.Module):
    def __init__(self,
                 output_feature_height=256,
                 output_feature_width=256,
                 image_h=1024,
                 image_w=1280,
                 patch_num_lat=10,
                 patch_num_lon=20,
                 patch_size=(0.3, 0.3),
                 patch_pixel_number=(64, 64),
                 crop_to_square=True,
                 camera_param_path=None,
                 train_cfg=None,
                 test_cfg=None,
                 ):
        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.input_feature_height = input_feature_height
        self.input_feature_width = input_feature_width

        self.crop_to_square = crop_to_square

        self.image_h = image_h
        self.image_w = image_w
        assert image_h <= image_w
        self.fisheye_param_path = camera_param_path

        self.fisheye_camera = FishEyeCameraCalibrated(self.fisheye_param_path)

        patches = self._create_sampling_patches(patch_num_lat, patch_num_lon, patch_size, patch_pixel_number)
        self.patches_2d = self._project_patches_to_original_image(patches)
        self.patches_2d = torch.from_numpy(self.patches_2d).float()
        self.patches_2d = torch.nn.Parameter(self.patches_2d, requires_grad=False)

        self.patch_center_image = self._create_2d_point_of_each_patch(patch_num_lat, patch_num_lon)
        self.patches_2d = torch.from_numpy(self.patches_2d).float()
        self.patches_2d = torch.nn.Parameter(self.patches_2d, requires_grad=False)


    def _create_2d_point_of_each_patch(self, patch_num_lat=10, patch_num_lon=20):
        lat_range = np.arange(1, patch_num_lat, 1)
        lon_range = np.arange(0, patch_num_lon, 1)
        lat_range = (lat_range / patch_num_lat) * np.pi / 2  # 0 ~ pi/2
        lon_range = ((lon_range / patch_num_lon) - 0.5) * 2 * np.pi  # -pi ~ pi

        patch_centers = np.dstack(np.meshgrid(lat_range, lon_range)).reshape(-1, 2)

        patch_center_3d = self.polar2cart(1, patch_centers[:, 0], patch_centers[:, 1]).T
        patch_center_image = self.fisheye_camera.world2camera(patch_center_3d)
        if self.crop_to_square:
            image_w_crop_left = (self.image_w - self.image_h) // 2
            patch_center_image[:, 0] -= image_w_crop_left
        patch_center_image = patch_center_image.astype(np.int32)
        return patch_center_image

    def polar2cart(self, r, lat, lon):
        return [
            r * math.sin(lat) * math.cos(lon),
            r * math.sin(lat) * math.sin(lon),
            r * math.cos(lat)
        ]



    def forward(self, input_feature_map):
        # sample from input feature map with the patch list
        # input feature map shape: (B, C, H, W)
        # patch 2d shape: (N_patches, H_patch, W_patch, 2)

        # first step: expand B batches to B * N_patches
        B, C, H, W = input_feature_map.shape
        N_patches, H_patch, W_patch, _ = self.patches_2d.shape
        input_feature_map = torch.unsqueeze(input_feature_map, dim=1)
        input_feature_map = torch.repeat_interleave(input_feature_map, N_patches, dim=1).view(B * N_patches, C, H, W)
        patches_2d = torch.unsqueeze(self.patches_2d, dim=0)
        patches_2d = torch.repeat_interleave(patches_2d, B, dim=0).view(B*N_patches, H_patch, W_patch, 2)
        sampled_patch = torch.nn.functional.grid_sample(input_feature_map, patches_2d, align_corners=True)
        sampled_patch = sampled_patch.view(B, N_patches, C, H_patch, W_patch).contiguous()
        return sampled_patch
