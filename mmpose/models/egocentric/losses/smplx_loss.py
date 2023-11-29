#  Copyright Jian Wang @ MPI-INF (c) 2023.
import torch
import torch.nn as nn

from mmpose.models.builder import LOSSES
from mmpose.models.utils.geometry import batch_rodrigues
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated


@LOSSES.register_module()
class FisheyeMeshLoss(nn.Module):
    """Mix loss for 3D human mesh. It is composed of loss on 2D joints, 3D
    joints, mesh vertices and smpl parameters (if any).

    Args:
        joints_2d_loss_weight (float): Weight for loss on 2D joints.
        joints_3d_loss_weight (float): Weight for loss on 3D joints.
        vertex_loss_weight (float): Weight for loss on 3D verteices.
        smpl_pose_loss_weight (float): Weight for loss on SMPL
            pose parameters.
        smpl_beta_loss_weight (float): Weight for loss on SMPL
            shape parameters.
        img_res (int): Input image resolution.
        focal_length (float): Focal length of camera model. Default=5000.
    """

    def __init__(self,
                 joints_2d_loss_weight,
                 joints_3d_loss_weight,
                 vertex_loss_weight,
                 smpl_pose_loss_weight,
                 smpl_beta_loss_weight,
                 smpl_beta_regularize_loss_weight,
                 camera_param_path):

        super().__init__()
        # Per-vertex loss on the mesh
        self.criterion_vertex = nn.L1Loss(reduction='none')

        # Joints (2D and 3D) loss
        self.criterion_joints_2d = nn.SmoothL1Loss(reduction='none')
        self.criterion_joints_3d = nn.SmoothL1Loss(reduction='none')

        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss(reduction='none')

        self.joints_2d_loss_weight = joints_2d_loss_weight
        self.joints_3d_loss_weight = joints_3d_loss_weight
        self.vertex_loss_weight = vertex_loss_weight
        self.smpl_pose_loss_weight = smpl_pose_loss_weight
        self.smpl_beta_loss_weight = smpl_beta_loss_weight
        self.smpl_beta_regularize_loss_weight = smpl_beta_regularize_loss_weight
        self.camera_param_path = camera_param_path
        self.fisheye_camera = FishEyeCameraCalibrated(self.camera_param_path)

    def joints_2d_loss(self, pred_joints_2d, gt_joints_2d, joints_2d_visible):
        """Compute 2D reprojection loss on the joints.

        The loss is weighted by joints_2d_visible.
        """
        conf = joints_2d_visible.float()
        conf = self.expand_conf_dim(conf, repeats=2)
        loss = (conf * self.criterion_joints_2d(pred_joints_2d, gt_joints_2d)).mean()
        return loss

    def expand_conf_dim(self, conf, repeats):
        if len(conf.shape) == 2:
            # in this case the conf shape is: (N_batch, N_joints)
            conf = torch.unsqueeze(conf, 2)
            conf = torch.repeat_interleave(conf, repeats=repeats, dim=2)
        return conf

    def joints_3d_loss(self, pred_joints_3d, gt_joints_3d, joints_3d_visible):
        """Compute 3D joints loss for the examples that 3D joint annotations
        are available.

        The loss is weighted by joints_3d_visible.
        """
        conf = joints_3d_visible.float()
        conf = self.expand_conf_dim(conf, repeats=3)
        if len(gt_joints_3d) > 0:
            # do not use the pelvis relative joint position
            # gt_pelvis = (gt_joints_3d[:, 2, :] + gt_joints_3d[:, 3, :]) / 2
            # gt_joints_3d = gt_joints_3d - gt_pelvis[:, None, :]
            # pred_pelvis = (pred_joints_3d[:, 2, :] +
            #                pred_joints_3d[:, 3, :]) / 2
            # pred_joints_3d = pred_joints_3d - pred_pelvis[:, None, :]
            return (
                conf *
                self.criterion_joints_3d(pred_joints_3d, gt_joints_3d)).mean()
        return pred_joints_3d.sum() * 0

    def vertex_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute 3D vertex loss for the examples that 3D human mesh
        annotations are available.

        The loss is weighted by the has_smpl.
        """
        conf = has_smpl.float()
        loss_vertex = self.criterion_vertex(pred_vertices, gt_vertices)
        loss_vertex = (conf[:, None, None] * loss_vertex).mean()
        return loss_vertex

    def smpl_losses(self, pred, gt, has_smpl):
        """Compute SMPL parameters loss for the examples that SMPL parameter
        annotations are available.

        The loss is weighted by has_smpl.
        """
        conf = has_smpl.float()
        loss = self.criterion_regr(pred, gt)
        loss = (conf[:, None] * loss).mean()
        return loss

    def project_points(self, points_3d, normalize=False):
        """Perform orthographic projection of 3D points using the camera
        parameters, return projected 2D points in image plane.

        Note:
            - batch size: B
            - point number: N

        Args:
            points_3d (Tensor([B, N, 3])): 3D points.
            normalize: if true, the result will be normalized to -1, 1

        Returns:
            Tensor([B, N, 2]): projected 2D points \
                in image space.
        """
        batch_size, joint_num, _ = points_3d.shape
        points_3d_flat = points_3d.view(batch_size * joint_num, 3)
        joints_2d = self.fisheye_camera.world2camera_pytorch(points_3d_flat, normalize=normalize)
        joints_2d = joints_2d.view(batch_size, joint_num, 2)
        return joints_2d

    def forward(self, output, target):
        """Forward function.

        Args:
            output (dict): dict of network predicted results.
                Keys: 'vertices', 'joints_3d', 'camera',
                'pose'(optional), 'beta'(optional)
            target (dict): dict of ground-truth labels.
                Keys: 'vertices', 'joints_3d', 'joints_3d_visible',
                'joints_2d', 'joints_2d_visible', 'pose', 'beta',
                'has_smpl'

        Returns:
            dict: dict of losses.
        """
        losses = {}
        has_smpl = target.get('has_smpl', None)
        # Per-vertex loss for the shape
        if 'vertices' in output.keys() and 'vertices' in target.keys():
            pred_vertices = output['vertices']

            gt_vertices = target['vertices']
            loss_vertex = self.vertex_loss(pred_vertices, gt_vertices, has_smpl)
            losses['vertex_loss'] = loss_vertex * self.vertex_loss_weight

        # Compute loss on SMPL parameters, if available
        if 'body_pose' in output.keys() and 'body_pose' in target.keys():
            loss_body_pose = self.smpl_losses(output['body_pose'], target['body_pose'], has_smpl)
            losses['smpl_pose_loss'] = loss_body_pose * self.smpl_pose_loss_weight
        if 'betas' in output.keys() and 'betas' in target.keys():
            loss_beta = self.smpl_losses(output['betas'], target['betas'], has_smpl)
            losses['smpl_beta_loss'] = loss_beta * self.smpl_beta_loss_weight
        if 'global_orient' in output.keys() and 'global_orient' in target.keys():
            loss_global_orient = self.smpl_losses(output['global_orient'], target['global_orient'], has_smpl)
            losses['smpl_global_orient_loss'] = loss_global_orient * self.smpl_global_orient_loss_weight
        if 'transl' in output.keys() and 'transl' in target.keys():
            loss_transl = self.smpl_losses(output['transl'], target['transl'], has_smpl)
            losses['smpl_transl_loss'] = loss_transl * self.smpl_transl_loss_weight

        # Compute 3D joints loss, always compute since we have 3d gt
        pred_joints_3d = output['keypoints_pred']
        gt_joints_3d = target['keypoints_3d']
        joints_3d_visible = target['keypoints_3d_visible']
        loss_joints_3d = self.joints_3d_loss(pred_joints_3d, gt_joints_3d,
                                             joints_3d_visible)
        losses['joints_3d_loss'] = loss_joints_3d * self.joints_3d_loss_weight

        if self.joints_2d_loss_weight > 0:
            # Compute 2D reprojection loss for the 2D joints
            gt_joints_2d = target['keypoints_2d']
            joints_2d_visible = target['keypoints_2d_visible']

            # pred joints 2d is in the image with height=1024, width=1280
            pred_joints_2d = self.project_points(pred_joints_3d, normalize=False)

            loss_joints_2d = self.joints_2d_loss(pred_joints_2d, gt_joints_2d,
                                                 joints_2d_visible)
            losses['joints_2d_loss'] = loss_joints_2d * self.joints_2d_loss_weight

        if self.smpl_beta_regularize_loss_weight > 0:
            # compute smplx beta regularization loss
            pred_smplx_beta = output['betas']
            loss_beta_regularize = torch.mean(torch.norm(pred_smplx_beta, dim=-1))
            losses['smpl_beta_regularize_loss'] = loss_beta_regularize * self.smpl_beta_regularize_loss_weight

        return losses
