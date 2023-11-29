#  Copyright Jian Wang @ MPI-INF (c) 2023.

import torch
import torch.nn as nn
from mmcv.ops.roi_align import roi_align
from torchvision.models.resnet import resnet18, ResNet18_Weights

from mmpose.models.builder import POSENETS
from mmpose.models.egocentric.iknet.network_utils import make_linear_layers
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)

@POSENETS.register_module()
class RefineNetMLP(nn.Module):
    def __init__(self,
                 body_feature_network_layers=(15 * (3 + 256 + 2), 512, 512, 15 * 3),
                 fisheye_calibration_file=None,
                 bbox_size=256,
                 image_feature_size=256,
                 joint_num=15,
                 add_joint_noise=False,
                 noise_std=0.001
                 ):
        """

        Args:
            joint_num: input 3d joint number
            smplx_config: smplx model setting, dict containing necessary information for initializing smplx model
        """

        super(RefineNetMLP, self).__init__()

        self.joint_num = joint_num
        self.image_feature_size = image_feature_size
        self.pose_refinenet = make_linear_layers(body_feature_network_layers, final_activation=False, use_bn=False,
                                                 activation_type='relu')

        self.fisheye_model = FishEyeCameraCalibrated(calibration_file_path=fisheye_calibration_file)
        self.image_encode_net = resnet18(weights=ResNet18_Weights.DEFAULT)
        # reset the fc layer
        self.image_encode_net.fc = nn.Linear(512, self.image_feature_size)
        self.bbox_size = bbox_size
        self.add_noise = add_joint_noise
        self.noise_std = noise_std

    def init_weights(self):
        for m in self.pose_refinenet:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm1d):
                constant_init(m, 1)


    def crop_image(self, original_image, keypoints_2d, bbox_size=224):
        # original image shape: (batch_size, 3, h, w)
        batch_size, _, h_img, w_img = original_image.shape
        # make crop bbox
        crop_bbox = torch.zeros(batch_size, self.joint_num, 4).to(keypoints_2d.device)
        # contains: (upleft_x, upleft_y, downright_x, downright_y)
        crop_bbox[:, :, :2] = keypoints_2d - bbox_size // 2
        crop_bbox[:, :, 2:] = keypoints_2d + bbox_size // 2

        padding_size = bbox_size

        original_image_pad = torch.nn.functional.pad(original_image,
                                                     pad=(padding_size, padding_size, padding_size, padding_size),
                                                     mode='constant',
                                                     value=0)
        crop_bbox_pad = crop_bbox + padding_size

        original_image_pad = torch.reshape(original_image_pad, (batch_size, 1, 3, h_img + 2 * padding_size,
                                                                w_img + 2 * padding_size))
        original_image_pad = torch.repeat_interleave(original_image_pad, repeats=self.joint_num,
                                                     dim=1)
        original_image_pad = torch.reshape(original_image_pad,
                                           (batch_size * self.joint_num, 3, h_img + 2 * padding_size,
                                            w_img + 2 * padding_size))

        crop_bbox_pad = torch.reshape(crop_bbox_pad, (batch_size * self.joint_num, 4))

        # combine batch index
        crop_bbox_pad = torch.cat(
            (torch.arange(crop_bbox_pad.shape[0]).float().to(crop_bbox_pad.device)[:, None], crop_bbox_pad),
            1)  # batch_idx, xmin, ymin, xmax, ymax
        cropped_image = roi_align(original_image_pad, crop_bbox_pad, (bbox_size, bbox_size),
                                  1.0, 0, 'avg', False)

        return cropped_image



    def forward(self, input):
        results = {}
        keypoints_3d = input['keypoints_3d']

        original_image = input['original_image'].float()
        batch_size = keypoints_3d.shape[0]

        # crop the joint image from the input
        # the keypoint is detached (maybe we can make the gradient flow back?), the learnable parameter can be the size?
        if torch.is_tensor(keypoints_3d):
            keypoints_3d = keypoints_3d.detach().to(original_image.device).float()
        else:
            keypoints_3d = torch.from_numpy(keypoints_3d).to(original_image.device).float()

        # if add noise to the keypoint 3d input
        results['keypoints_3d_wo_noise'] = keypoints_3d.clone()
        if self.add_noise:
            noise = self.noise_std ** 0.5 * torch.randn(keypoints_3d.shape).to(original_image.device)
            distance = torch.linalg.norm(keypoints_3d, dim=-1)[:, :, None]
            noise = noise * (distance ** 0.5)  # add noise to different distance to the camera
            keypoints_3d += noise

        # project back to keypoints 2d
        keypoints_2d_proj = self.fisheye_model.world2camera_pytorch(keypoints_3d.view(-1, 3)).view(-1,
                                                                                                   self.joint_num,
                                                                                                   2).float()

        cropped_image = self.crop_image(original_image, keypoints_2d_proj, bbox_size=self.bbox_size)
        cropped_image = cropped_image.view(batch_size * self.joint_num, 3, self.bbox_size, self.bbox_size)
        # cropped_image = self.image_encode_net_transform(cropped_image)
        cropped_image_feature = self.image_encode_net(cropped_image)
        cropped_image_feature = cropped_image_feature.reshape(batch_size, self.joint_num * self.image_feature_size)

        network_input = torch.cat(
            [
                cropped_image_feature,
                keypoints_3d.reshape(batch_size, self.joint_num * 3),
                keypoints_2d_proj.reshape(batch_size, self.joint_num * 2)
            ], dim=-1
        )

        keypoints_3d_refined = self.pose_refinenet(network_input)
        keypoints_3d_refined = keypoints_3d_refined.view(batch_size, self.joint_num, 3)
        results['keypoints_pred'] = keypoints_3d_refined
        results['keypoints_3d_w_noise'] = keypoints_3d
        return results

    def debug_crop_img(self, input):
        keypoints_3d = input['keypoints_3d']
        original_image = input['original_image']
        batch_size = keypoints_3d.shape[0]

        # crop the joint image from the input
        # the keypoint is detached (maybe we can make the gradient flow back?), the learnable parameter can be the size?
        keypoints_3d_torch = keypoints_3d.detach()

        # if add noise to the keypoint 3d input
        if self.add_noise:
            noise = self.noise_std ** 0.5 * torch.randn(keypoints_3d.shape)
            distance = torch.linalg.norm(keypoints_3d, dim=-1)[:, :, None]
            noise = noise * distance  # add noise to different distance to the camera
            keypoints_3d += noise

        # project back to keypoints 2d
        keypoints_2d_proj = self.fisheye_model.world2camera_pytorch(keypoints_3d_torch.view(-1, 3)).view(-1,
                                                                                                         self.joint_num,
                                                                                                         2)
        cropped_image = self.crop_image(original_image, keypoints_2d_proj, bbox_size=self.bbox_size)
        return cropped_image
