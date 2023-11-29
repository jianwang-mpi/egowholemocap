import numpy as np
from copy import copy
from copy import copy

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate

# from utils import op, multiview, img, misc, volumetric
from mmpose.models.egocentric.ablation.scene_ego_utils import op
# from network import pose_resnet
# from network.v2v import V2VModel, V2VModelSimple
from mmpose.models.egocentric.ablation.v2v import V2VModel
# from utils.fisheye.FishEyeCalibrated import FishEyeCameraCalibrated
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated


# from utils_proj.fisheye.FishEyeCalibrated import FishEyeCameraCalibrated
# from utils_proj.fisheye.FishEyeEquisolid import FishEyeCameraEquisolid


class VoxelNetwork(nn.Module):
    def __init__(self, batch_size,
                 camera_calibration_file_path,
                 device='cuda'
                 ):
        super(VoxelNetwork, self).__init__()

        self.device = device
        self.num_joints = 15

        # volume
        self.volume_softmax = True
        self.volume_multiplier = 1
        self.volume_size = 64

        self.cuboid_side = 3

        self.kind = 'mo2cap2'

        # heatmap
        self.heatmap_softmax = True
        self.heatmap_multiplier = 100.0

        # resize and pad feature for reprojection
        self.process_features = nn.Sequential(
            nn.Conv2d(768, 32, 1),
            nn.Upsample(size=(1024, 1024)),
            nn.ConstantPad2d(padding=(128, 128, 0, 0), value=0.0)
        )
        self.process_features = self.process_features.to(device)

        self.with_scene = False
        volume_input_channel_num = 32

        self.volume_net = V2VModel(volume_input_channel_num, self.num_joints)
        self.volume_net = self.volume_net.to(device)

        print('build coord volume')
        self.coord_volume = self.build_coord_volume()
        self.coord_volumes = self.coord_volume.unsqueeze(0).expand(batch_size,
                                                                   -1, -1, -1, -1)
        self.coord_volumes = self.coord_volumes.to(device)

        self.fisheye_camera_model = FishEyeCameraCalibrated(
            calibration_file_path=camera_calibration_file_path)
        print('build reprojected grid coord')
        self.grid_coord_proj = op.get_projected_2d_points_with_coord_volumes(fisheye_model=self.fisheye_camera_model,
                                                                             coord_volume=self.coord_volume)
        self.grid_coord_proj.requires_grad = False

        self.grid_coord_proj_batch = op.get_grid_coord_proj_batch(self.grid_coord_proj,
                                                                  batch_size=batch_size,
                                                                  heatmap_shape=(1024, 1280))
        self.grid_coord_proj_batch.requires_grad = False
        self.grid_coord_proj_batch = self.grid_coord_proj_batch.to(device)

        self.ray = self.calculated_ray_direction_numpy(1280,
                                                       1024)

        self.image_width = 1280
        self.image_height = 1024

    def build_coord_volume(self):
        """
        get coord volume and prepare for the re-projection process
        :param self:
        :return:
        """
        # build coord volumes
        sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])

        position = np.array([-self.cuboid_side / 2, -self.cuboid_side / 2, 0])
        # build coord volume
        xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size),
                                       torch.arange(self.volume_size),
                                       torch.arange(self.volume_size))
        grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
        grid = grid.reshape((-1, 3))

        grid_coord = torch.zeros_like(grid)
        grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
        grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
        grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

        coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

        return coord_volume

    def calculated_ray_direction(self, image_width, image_height):
        points = np.zeros(shape=(image_width, image_height, 2))
        x_range = np.array(range(image_width))
        y_range = np.array(range(image_height))
        points[:, :, 0] = np.add(points[:, :, 0].transpose(), x_range).transpose()
        points[:, :, 1] = np.add(points[:, :, 1], y_range)
        points = points.reshape((-1, 2))
        ray = self.fisheye_camera_model.camera2world_ray(points)
        ray_torch = torch.from_numpy(ray)
        return ray_torch

    def calculated_ray_direction_numpy(self, image_width, image_height):
        points = np.zeros(shape=(image_width, image_height, 2))
        x_range = np.array(range(image_width))
        y_range = np.array(range(image_height))
        points[:, :, 0] = np.add(points[:, :, 0].transpose(), x_range).transpose()
        points[:, :, 1] = np.add(points[:, :, 1], y_range)
        points = points.reshape((-1, 2))
        ray = self.fisheye_camera_model.camera2world_ray(points)
        return ray


    def forward(self, features):
        # process features before unprojecting
        features = self.process_features(features)

        # lift to volume
        if features.shape[0] < self.grid_coord_proj_batch.shape[0]:
            grid_coord_proj_batch_input = self.grid_coord_proj_batch[:features.shape[0]]
        else:
            grid_coord_proj_batch_input = self.grid_coord_proj_batch
        volumes = op.unproject_heatmaps_one_view_batch(features, grid_coord_proj_batch_input, self.volume_size)

        # integral 3d
        volumes = self.volume_net(volumes)
        if volumes.shape[0] < self.coord_volumes.shape[0]:
            coord_volumes_input = self.coord_volumes[:volumes.shape[0]]
        else:
            coord_volumes_input = self.coord_volumes
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                            coord_volumes_input,
                                                                            softmax=self.volume_softmax)

        return vol_keypoints_3d
