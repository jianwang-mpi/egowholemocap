import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.models.egocentric.ablation.scene_ego_utils.img import to_numpy, to_torch
from mmpose.models.egocentric.ablation.scene_ego_utils import multiview


def integrate_tensor_2d(heatmaps, softmax=True):
    """Applies softmax to heatmaps and integrates them to get their's "center of masses"

    Args:
        heatmaps torch tensor of shape (batch_size, n_heatmaps, h, w): input heatmaps

    Returns:
        coordinates torch tensor of shape (batch_size, n_heatmaps, 2): coordinates of center of masses of all heatmaps

    """
    batch_size, n_heatmaps, h, w = heatmaps.shape

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, -1))
    if softmax:
        heatmaps = nn.functional.softmax(heatmaps, dim=2)
    else:
        heatmaps = nn.functional.relu(heatmaps)

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, h, w))

    mass_x = heatmaps.sum(dim=2)
    mass_y = heatmaps.sum(dim=3)

    mass_times_coord_x = mass_x * torch.arange(w).type(torch.float).to(mass_x.device)
    mass_times_coord_y = mass_y * torch.arange(h).type(torch.float).to(mass_y.device)

    x = mass_times_coord_x.sum(dim=2, keepdim=True)
    y = mass_times_coord_y.sum(dim=2, keepdim=True)

    if not softmax:
        x = x / mass_x.sum(dim=2, keepdim=True)
        y = y / mass_y.sum(dim=2, keepdim=True)

    coordinates = torch.cat((x, y), dim=2)
    coordinates = coordinates.reshape((batch_size, n_heatmaps, 2))

    return coordinates, heatmaps


def integrate_tensor_3d(volumes, softmax=True):
    batch_size, n_volumes, x_size, y_size, z_size = volumes.shape

    volumes = volumes.reshape((batch_size, n_volumes, -1))
    if softmax:
        volumes = nn.functional.softmax(volumes, dim=2)
    else:
        volumes = nn.functional.relu(volumes)

    volumes = volumes.reshape((batch_size, n_volumes, x_size, y_size, z_size))

    mass_x = volumes.sum(dim=3).sum(dim=3)
    mass_y = volumes.sum(dim=2).sum(dim=3)
    mass_z = volumes.sum(dim=2).sum(dim=2)

    mass_times_coord_x = mass_x * torch.arange(x_size).type(torch.float).to(mass_x.device)
    mass_times_coord_y = mass_y * torch.arange(y_size).type(torch.float).to(mass_y.device)
    mass_times_coord_z = mass_z * torch.arange(z_size).type(torch.float).to(mass_z.device)

    x = mass_times_coord_x.sum(dim=2, keepdim=True)
    y = mass_times_coord_y.sum(dim=2, keepdim=True)
    z = mass_times_coord_z.sum(dim=2, keepdim=True)

    if not softmax:
        x = x / mass_x.sum(dim=2, keepdim=True)
        y = y / mass_y.sum(dim=2, keepdim=True)
        z = z / mass_z.sum(dim=2, keepdim=True)

    coordinates = torch.cat((x, y, z), dim=2)
    coordinates = coordinates.reshape((batch_size, n_volumes, 3))

    return coordinates, volumes


def integrate_tensor_3d_with_coordinates(volumes, coord_volumes, softmax=True):

    batch_size, n_volumes, x_size, y_size, z_size = volumes.shape
    volumes = volumes.reshape((batch_size, n_volumes, -1))
    if softmax:
        volumes = nn.functional.softmax(volumes, dim=2)
    else:
        # need to be normalized
        volumes = nn.functional.relu(volumes)

    volumes = volumes.reshape((batch_size, n_volumes, x_size, y_size, z_size))
    coordinates = torch.einsum("bnxyz, bxyzc -> bnc", volumes, coord_volumes)

    return coordinates, volumes

def get_projected_2d_points_with_coord_volumes(fisheye_model, coord_volume):
    """
    :param fisheye_model:
    :param coord_volumes: no batch dimension
    :return:
    """
    # Note: coord volumes are the same among all of the batches, so we only need to
    # get the coord volume for one batch and copy it to others

    device = coord_volume.device
    volume_shape = coord_volume.shape  # x_len, y_len, z_len

    grid_coord = coord_volume.reshape((-1, 3))

    ####note: precalculated reprojected points!
    grid_coord_proj = multiview.project_3d_points_to_image_fisheye_camera(
        fisheye_model, grid_coord
    )
    return grid_coord_proj


def get_distance_with_coord_volumes(coord_volume):
    """
    :param fisheye_model:
    :param coord_volumes: no batch dimension
    :return:
    """
    # Note: coord volumes are the same among all of the batches, so we only need to
    # get the coord volume for one batch and copy it to others

    grid_coord = coord_volume.reshape((-1, 3))

    ####note: precalculated distance!
    grid_coord_distance = torch.norm(grid_coord, dim=-1)
    return grid_coord_distance


def unproject_heatmaps_one_view(heatmaps, grid_coord_proj, volume_size):

    '''
    project the heatmap based on the camera parameters of egocentric fisheye camera
    :param heatmaps:
    :param fisheye_model: fisheye camera model
    :param coord_volumes: shape: batch_size, n_joints, x_len, y_len, z_len
    :return:
    '''
    # Note: the coord volume is the same for all images, thus we can calculate it in advance.
    #  We do not need to calculate
    #  it within the iteration
    device = heatmaps.device
    batch_size, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], tuple(heatmaps.shape[2:])
    volume_shape = (volume_size, volume_size, volume_size)

    volume_batch = torch.zeros(batch_size, n_joints, *volume_shape, device=device)

    # TODO: speed up this this loop
    for batch_i in range(batch_size):
        heatmap = heatmaps[batch_i]
        heatmap = heatmap.unsqueeze(0)

        # transform to [-1.0, 1.0] range
        # note: in grid_coord_proj, the format is like (x, y), however,
        # note: when we sample the points, we need (y, x)
        grid_coord_proj_transformed = torch.zeros_like(grid_coord_proj)
        grid_coord_proj_transformed[:, 0] = 2 * (grid_coord_proj[:, 0] / heatmap_shape[1] - 0.5)
        grid_coord_proj_transformed[:, 1] = 2 * (grid_coord_proj[:, 1] / heatmap_shape[0] - 0.5)

        # prepare to F.grid_sample
        grid_coord_proj_transformed = grid_coord_proj_transformed.unsqueeze(1).unsqueeze(0)

        current_volume = F.grid_sample(heatmap, grid_coord_proj_transformed, align_corners=True)

        # reshape back to volume
        current_volume = current_volume.view(n_joints, *volume_shape)

        volume_batch[batch_i] = current_volume

    return volume_batch

def get_grid_coord_proj_batch(grid_coord_proj, batch_size, heatmap_shape):
    grid_coord_proj_transformed = torch.zeros_like(grid_coord_proj)
    grid_coord_proj_transformed[:, 0] = 2 * (grid_coord_proj[:, 0] / heatmap_shape[1] - 0.5)
    grid_coord_proj_transformed[:, 1] = 2 * (grid_coord_proj[:, 1] / heatmap_shape[0] - 0.5)
    grid_coord_proj_transformed = grid_coord_proj_transformed.unsqueeze(1).unsqueeze(0)
    grid_coord_proj_transformed_batch = grid_coord_proj_transformed.expand(batch_size, -1, -1, -1)

    return grid_coord_proj_transformed_batch


def get_grid_coord_distance_batch(grid_coord_distance, batch_size, joint_num=15):
    grid_coord_distance = grid_coord_distance.unsqueeze(1).unsqueeze(0)
    grid_coord_distance_batch = grid_coord_distance.expand(batch_size, joint_num, -1, -1)

    return grid_coord_distance_batch


def unproject_heatmaps_one_view_batch(heatmaps, grid_coord_proj_transformed_batch, volume_size):

    '''
    project the heatmap based on the camera parameters of egocentric fisheye camera
    :param heatmaps:
    :param fisheye_model: fisheye camera model
    :param coord_volumes: shape: batch_size, n_joints, x_len, y_len, z_len
    :return:
    '''
    # Note: the coord volume is the same for all images, thus we can calculate it in advance.
    #  We do not need to calculate
    #  it within the iteration
    batch_size, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], tuple(heatmaps.shape[2:])
    volume_shape = (volume_size, volume_size, volume_size)

    current_volume = F.grid_sample(heatmaps, grid_coord_proj_transformed_batch, align_corners=True)

    # reshape back to volume
    volume_batch = current_volume.view(batch_size, n_joints, *volume_shape)

    return volume_batch

def gaussian_2d_pdf(coords, means, sigmas, normalize=True):
    normalization = 1.0
    if normalize:
        normalization = (2 * np.pi * sigmas[:, 0] * sigmas[:, 0])

    exp = torch.exp(-((coords[:, 0] - means[:, 0]) ** 2 / sigmas[:, 0] ** 2 + (coords[:, 1] - means[:, 1]) ** 2 / sigmas[:, 1] ** 2) / 2)
    return exp / normalization


def render_points_as_2d_gaussians(points, sigmas, image_shape, normalize=True):
    device = points.device
    n_points = points.shape[0]

    yy, xx = torch.meshgrid(torch.arange(image_shape[0]).to(device), torch.arange(image_shape[1]).to(device))
    grid = torch.stack([xx, yy], dim=-1).type(torch.float32)
    grid = grid.unsqueeze(0).repeat(n_points, 1, 1, 1)  # (n_points, h, w, 2)
    grid = grid.reshape((-1, 2))

    points = points.unsqueeze(1).unsqueeze(1).repeat(1, image_shape[0], image_shape[1], 1)
    points = points.reshape(-1, 2)

    sigmas = sigmas.unsqueeze(1).unsqueeze(1).repeat(1, image_shape[0], image_shape[1], 1)
    sigmas = sigmas.reshape(-1, 2)

    images = gaussian_2d_pdf(grid, points, sigmas, normalize=normalize)
    images = images.reshape(n_points, *image_shape)

    return images
