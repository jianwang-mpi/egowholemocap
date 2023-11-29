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
from mmpose.models import builder
from mmpose.models.builder import POSENETS
from mmpose.models.detectors.base import BasePose
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated
from scipy.spatial.transform import Rotation as R
import open3d
import torch

@POSENETS.register_module()
class PosePatchGenerator(nn.Module):
    def __init__(self,
                 input_feature_height=256,
                 input_feature_width=256,
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


    def _generate_patch_coordinates(self, patch_center_lat, patch_center_lon, patch_size, patch_pixel_num, visualize=False):
        # generate (x, y) pairs
        x_range = np.arange(0, patch_pixel_num[0]) / (patch_pixel_num[0] - 1) - 0.5
        y_range = np.arange(0, patch_pixel_num[1]) / (patch_pixel_num[1] - 1) - 0.5

        x_range *= 2
        y_range *= 2

        x_range *= patch_size[0]
        y_range *= patch_size[1]

        patch_centers = np.dstack(np.meshgrid(x_range, y_range)).reshape(-1, 2)
        patch_centers = np.concatenate([patch_centers, np.ones((len(patch_centers), 1))], axis=1)
        r_lat = R.from_euler('xyz', (patch_center_lat, 0, 0), degrees=False)
        r_lon = R.from_euler('xyz', (0, 0, patch_center_lon), degrees=False)
        rotated_patch_centers = r_lat.apply(patch_centers)
        rotated_patch_centers = r_lon.apply(rotated_patch_centers)

        if visualize:
            sphere_list = []
            coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
            for patch_center in rotated_patch_centers:
                sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(patch_center)
                sphere_list.append(sphere)
            open3d.visualization.draw_geometries(sphere_list + [coord])

        rotated_patch_centers = rotated_patch_centers.reshape((len(x_range), len(y_range), 3))
        return rotated_patch_centers

    def _create_sampling_patches(self, patch_num_lat=10, patch_num_lon=20,
                                 patch_size=(0.3, 0.3), patch_pixel_number=(64, 64)):
        """
        Generate sampling patches given image size,
        Args:
            patch_num_lat:
            patch_num_lon:
            patch_size:
            patch_pixel_number:

        Returns:

        """
        lat_range = np.arange(1, patch_num_lat, 1)
        lon_range = np.arange(0, patch_num_lon, 1)
        lat_range = (lat_range / patch_num_lat) * np.pi / 2  # 0 ~ pi/2
        lon_range = ((lon_range / patch_num_lon) - 0.5) * 2 * np.pi  # -pi ~ pi

        patch_centers = np.dstack(np.meshgrid(lat_range, lon_range)).reshape(-1, 2)

        # patch_center_cart_list = []
        patch_coordinate_list = []
        for i, patch_center in enumerate(patch_centers):
            coordinate_path_center = self.polar2cart(1, patch_center[0], patch_center[1])
            # patch_center_cart_list.append(coordinate_path_center)

            patch_coordinates = self._generate_patch_coordinates(patch_center[0], patch_center[1],
                                                                patch_size, patch_pixel_number)
            patch_coordinate_list.append(patch_coordinates)
        return patch_coordinate_list

    def _project_patches_to_original_image(self, patch_coordinate_list):
        patch_on_image_list = []
        for patch in patch_coordinate_list:
            H, W, _ = patch.shape
            patch_flat = patch.reshape((-1, 3))
            pos_2d_flat = self.fisheye_camera.world2camera(patch_flat)
            pose_2d_list = pos_2d_flat.reshape((H, W, 2))
            # in patch pixel position (H, W), the first dimension of pose_2d is x position, second dimension is y
            # position
            if self.crop_to_square:
                image_w_crop_left = (self.image_w - self.image_h) // 2
                # image_w_crop_right = (self.image_w - self.image_h) // 2
                pose_2d_list[:, :, 0] -= image_w_crop_left
            # resize to [-1, 1]
            pose_2d_list = (pose_2d_list - self.image_h / 2) / (self.image_h / 2)

            patch_on_image_list.append(pose_2d_list)
        patch_on_image_list = np.asarray(patch_on_image_list)  # shape: (sample_number, H, W, 2)
        return patch_on_image_list

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
