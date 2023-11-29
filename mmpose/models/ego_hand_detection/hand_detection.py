#  Copyright Jian Wang @ MPI-INF (c) 2023.

import cv2
import numpy as np

from mmpose.models import build_posenet
from mmpose.models import POSENETS

import torch
import torch.nn as nn

from mmdet.apis import init_detector
from mmdet.apis import inference_detector
# from mmpose.models.egocentric_hand.ego_hand_detection.mmdet_inference import inference_detector_img_tensor
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated
from mmcv.ops.roi_align import roi_align

@POSENETS.register_module()
class HandDetection(nn.Module):
    def __init__(self, det_config, det_checkpoint, fisheye_calibration_file,
                 crop_size=(256, 256)):
        super(HandDetection, self).__init__()

        self.det_model = init_detector(det_config, det_checkpoint)
        self.fisheye_camera = FishEyeCameraCalibrated(fisheye_calibration_file)

        self.crop_size = crop_size


    def get_bbox(self, hand_joint):
        # return: left_top_x, left_top_y, right_bottom_x, right_bottom_y
        crop_size = torch.asarray(self.crop_size)
        left_top_corner = hand_joint - crop_size / 2
        right_bottom_corner = hand_joint + crop_size / 2
        return torch.cat([left_top_corner, right_bottom_corner], dim=-1)


    def get_hand_bbox(self, keypoints_2d):
        # return: left hand crop, right hand crop

        # create a bounding box for the hand
        B, N, _ = keypoints_2d.shape
        assert N == 15 and _ == 2
        right_hand_joint = keypoints_2d[:, 3, :]
        left_hand_joint = keypoints_2d[:, 6, :]

        # get the bounding box
        right_hand_bbox = self.get_bbox(right_hand_joint)
        left_hand_bbox = self.get_bbox(left_hand_joint)
        return left_hand_bbox, right_hand_bbox

    def crop_hand_image(self, raw_img, crop_bbox):
        raw_img = raw_img.clone()
        padding_size = self.crop_size[0]
        assert self.crop_size[0] == self.crop_size[1]
        original_image_pad = torch.nn.functional.pad(raw_img,
                                                     pad=(self.crop_size[0], self.crop_size[0],
                                                          self.crop_size[1], self.crop_size[1]),
                                                     mode='constant',
                                                     value=0)
        crop_bbox_pad = crop_bbox + padding_size

        # combine batch index
        crop_bbox_pad = torch.cat(
            (torch.arange(crop_bbox_pad.shape[0]).float().to(crop_bbox_pad.device)[:, None], crop_bbox_pad),
            1)  # batch_idx, xmin, ymin, xmax, ymax
        cropped_image = roi_align(original_image_pad, crop_bbox_pad, self.crop_size,
                                  1.0, 0, 'avg', False)
        return cropped_image

    def recover_bbox(self, small_bbox, large_bbox):
        # small_bbox: left_top_x, left_top_y, right_bottom_x, right_bottom_y
        # large_bbox: left_top_x, left_top_y, right_bottom_x, right_bottom_y
        # return: left_top_x, left_top_y, right_bottom_x, right_bottom_y
        left_top_x = small_bbox[:, 0] + large_bbox[:, 0]
        left_top_y = small_bbox[:, 1] + large_bbox[:, 1]
        right_bottom_x = small_bbox[:, 2] + large_bbox[:, 0]
        right_bottom_y = small_bbox[:, 3] + large_bbox[:, 1]
        return torch.cat([left_top_x, left_top_y, right_bottom_x, right_bottom_y], dim=-1)


    def tensor_batch_to_numpy(self, image_batch):
        # image_batch: B, C, H, W
        # return: B, H, W, C

        images = image_batch.cpu().numpy()

        images *= np.asarray([0.229, 0.224, 0.225])[None, :, None, None]
        images += np.asarray([0.485, 0.456, 0.406])[None, :, None, None]

        images = np.transpose(images, (0, 2, 3, 1)) * 255
        images = images.astype(np.uint8)[:, :, :, ::-1]

        images = [images[i] for i in range(images.shape[0])]
        return images

    def forward(self, input_dict):
        raw_img = input_dict['original_image']
        keypoints_3d = input_dict['keypoints_3d']

        #project the keypoints 3d to 2d
        B, N, _ = keypoints_3d.shape
        keypoints_3d = keypoints_3d.view(B*N, 3)
        keypoints_2d = self.fisheye_camera.world2camera_pytorch(keypoints_3d)
        keypoints_2d = keypoints_2d.view(B, N, 2)

        #crop the rough image based on the 2d keypoints
        left_hand_bbox, right_hand_bbox = self.get_hand_bbox(keypoints_2d)
        left_hand_crop = self.crop_hand_image(raw_img, left_hand_bbox)
        right_hand_crop = self.crop_hand_image(raw_img, right_hand_bbox)

        # detect the left hand
        left_hand_crop = self.tensor_batch_to_numpy(left_hand_crop)
        # save the left hand crop
        for image_idx, image in enumerate(left_hand_crop):
            cv2.imwrite('/CT/EgoMocap/work/EgocentricFullBody/vis_results/left_hand_crop_{}.png'.format(image_idx), image)
        left_hand_result = inference_detector(self.det_model, left_hand_crop)
        print(left_hand_result)


        # detect the right hand
        right_hand_crop = self.tensor_batch_to_numpy(right_hand_crop)
        for image_idx, image in enumerate(right_hand_crop):
            cv2.imwrite('/CT/EgoMocap/work/EgocentricFullBody/vis_results/right_hand_crop_{}.png'.format(image_idx),
                        image)
        right_hand_result = inference_detector(self.det_model, right_hand_crop)
        print(right_hand_result)


        left_hand_final_bbox = self.recover_bbox(left_hand_small_bbox, left_hand_bbox)
        right_hand_final_bbox = self.recover_bbox(right_hand_small_bbox, right_hand_bbox)

        result_dict = {
            'left_hand_bbox': left_hand_final_bbox,
            'right_hand_bbox': right_hand_final_bbox
        }
        return result_dict


