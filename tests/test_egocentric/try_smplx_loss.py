# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from numpy.testing import assert_almost_equal

from mmpose.models import build_loss
from mmpose.models.utils.geometry import batch_rodrigues


def try_mesh_loss():
    """test mesh loss."""
    loss_cfg = dict(
        type='FisheyeMeshLoss',
        joints_2d_loss_weight=1,
        joints_3d_loss_weight=1,
        vertex_loss_weight=1,
        smpl_pose_loss_weight=1,
        smpl_beta_loss_weight=1,
        camera_param_path=r'Z:\EgoMocap\work\EgocentricFullBody\mmpose\utils\fisheye_camera\fisheye.calibration_01_12.json')

    loss = build_loss(loss_cfg)

    smpl_pose = torch.zeros([1, 72], dtype=torch.float32)
    smpl_rotmat = batch_rodrigues(smpl_pose.view(-1, 3)).view(-1, 24, 3, 3)
    smpl_beta = torch.zeros([1, 10], dtype=torch.float32)
    vertices = torch.rand([1, 6890, 3], dtype=torch.float32)
    joints_3d = torch.ones([1, 15, 3], dtype=torch.float32)
    joints_2d = loss.project_points(joints_3d, normalize=False)

    fake_pred = {}
    fake_pred['pose'] = smpl_rotmat
    fake_pred['beta'] = smpl_beta
    fake_pred['vertices'] = vertices
    fake_pred['joints_3d'] = joints_3d

    fake_gt = {}
    # fake_gt['pose'] = smpl_pose
    # fake_gt['beta'] = smpl_beta
    # fake_gt['vertices'] = vertices
    fake_gt['has_smpl'] = torch.zeros(1, dtype=torch.float32)
    fake_gt['joints_3d'] = joints_3d
    fake_gt['joints_3d_visible'] = torch.ones([1, 15, 1], dtype=torch.float32)
    fake_gt['joints_2d'] = joints_2d
    fake_gt['joints_2d_visible'] = torch.ones([1, 15, 1], dtype=torch.float32)

    losses = loss(fake_pred, fake_gt)
    # assert torch.allclose(losses['vertex_loss'], torch.tensor(0.))
    # assert torch.allclose(losses['smpl_pose_loss'], torch.tensor(0.))
    # assert torch.allclose(losses['smpl_beta_loss'], torch.tensor(0.))
    assert torch.allclose(losses['joints_3d_loss'], torch.tensor(0.))
    assert torch.allclose(losses['joints_2d_loss'], torch.tensor(0.))

    # fake_pred = {}
    # fake_pred['pose'] = smpl_rotmat + 1
    # fake_pred['beta'] = smpl_beta + 1
    # fake_pred['vertices'] = vertices + 1
    # fake_pred['joints_3d'] = joints_3d.clone()
    #
    # joints_3d_t = joints_3d.clone()
    # joints_3d_t[:, 0] = joints_3d_t[:, 0] + 1
    # fake_gt = {}
    # fake_gt['pose'] = smpl_pose
    # fake_gt['beta'] = smpl_beta
    # fake_gt['vertices'] = vertices
    # fake_gt['has_smpl'] = torch.ones(1, dtype=torch.float32)
    # fake_gt['joints_3d'] = joints_3d_t
    # fake_gt['joints_3d_visible'] = torch.ones([1, 15, 1], dtype=torch.float32)
    # fake_gt['joints_2d'] = joints_2d + 128
    # fake_gt['joints_2d_visible'] = torch.ones([1, 15, 1], dtype=torch.float32)
    #
    # losses = loss(fake_pred, fake_gt)
    # assert torch.allclose(losses['vertex_loss'], torch.tensor(1.))
    # assert torch.allclose(losses['smpl_pose_loss'], torch.tensor(1.))
    # assert torch.allclose(losses['smpl_beta_loss'], torch.tensor(1.))
    # assert torch.allclose(losses['joints_3d_loss'], torch.tensor(0.5 / 15))
    # assert torch.allclose(losses['joints_2d_loss'], torch.tensor(128. - 0.5))  # smooth l1 loss

if __name__ == '__main__':
    try_mesh_loss()