import cv2
import numpy as np
from copy import copy, deepcopy

import open3d

from mmpose.datasets.builder import PIPELINES
from mmpose.utils.visualization.skeleton import Skeleton
import math
import torch
from torchvision.transforms import functional as F
import torchvision.transforms as transforms


@PIPELINES.register_module()
class Generate2DHandPose:
    def __init__(self, fisheye_model_path):
        super().__init__()
        self.skeleton_model = Skeleton(fisheye_model_path)

    def _generate_single_hand_pose(self, hand_keypoints_3d, hand_keypoints_3d_visible):
        assert np.sum(hand_keypoints_3d_visible) == 21  # assert all joints are visible
        hand_pose_2d = np.zeros((hand_keypoints_3d.shape[0], 2))
        hand_keypoints_3d_vis = hand_keypoints_3d[hand_keypoints_3d_visible > 0]
        hand_pose_2d_vis = self.skeleton_model.camera.world2camera(hand_keypoints_3d_vis)
        hand_pose_2d[hand_keypoints_3d_visible > 0] = hand_pose_2d_vis
        return hand_pose_2d

    def __call__(self, results: dict) -> dict:
        left_hand_keypoints_3d = results['left_hand_keypoints_3d']
        left_hand_keypoints_3d_visible = results['left_hand_keypoints_3d_visible']
        left_hand_keypoints_2d = self._generate_single_hand_pose(left_hand_keypoints_3d, left_hand_keypoints_3d_visible)
        results['left_hand_keypoints_2d'] = left_hand_keypoints_2d
        results['left_hand_keypoints_2d_visible'] = left_hand_keypoints_3d_visible

        right_hand_keypoints_3d = results['right_hand_keypoints_3d']
        right_hand_keypoints_3d_visible = results['right_hand_keypoints_3d_visible']
        right_hand_keypoints_2d = self._generate_single_hand_pose(right_hand_keypoints_3d, right_hand_keypoints_3d_visible)
        results['right_hand_keypoints_2d'] = right_hand_keypoints_2d
        results['right_hand_keypoints_2d_visible'] = right_hand_keypoints_3d_visible
        return results



@PIPELINES.register_module()
class CropHandImage:
    def __init__(self, enlarge_scale=1.2, padding_size=256):
        super().__init__()
        self.enlarge_scale = enlarge_scale
        self.padding_size = padding_size


    def _create_hand_bbox(self, hand_keypoints_2d, enlarge_scale):
        # hand_keypoints_2d: 21 x 2
        # hand_center: 2

        hand_bbox = np.zeros(4)
        hand_bbox[0: 2] = np.min(hand_keypoints_2d, axis=0)
        hand_bbox[2: 4] = np.max(hand_keypoints_2d, axis=0)

        bbox_w = hand_bbox[2] - hand_bbox[0]
        bbox_h = hand_bbox[3] - hand_bbox[1]
        bbox_center = np.array([hand_bbox[0] + bbox_w / 2, hand_bbox[1] + bbox_h / 2])
        bbox_size = max(bbox_w, bbox_h) * enlarge_scale
        hand_bbox[0:2] = bbox_center - bbox_size / 2
        hand_bbox[2:4] = bbox_center + bbox_size / 2

        return hand_bbox

    def _crop_hand(self, img, hand_bbox, padding_size):
        # hand_bbox: 4
        # first pad the image
        img_pad = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), 'constant')
        hand_bbox += padding_size
        hand_bbox = hand_bbox.astype(np.int32)
        hand_img = img_pad[hand_bbox[1]:hand_bbox[3], hand_bbox[0]:hand_bbox[2]]
        return hand_img

    def __call__(self, results: dict) -> dict:
        img = results['img']
        img_h, img_w, _ = img.shape
        left_hand_keypoints_2d = results['left_hand_keypoints_2d']
        right_hand_keypoints_2d = results['right_hand_keypoints_2d']
        left_hand_bbox = self._create_hand_bbox(left_hand_keypoints_2d, self.enlarge_scale)
        right_hand_bbox = self._create_hand_bbox(right_hand_keypoints_2d, self.enlarge_scale)

        results['left_hand_bbox'] = left_hand_bbox
        results['right_hand_bbox'] = right_hand_bbox

        # the 2d joints should also be treated here

        left_hand_keypoints_2d = results['left_hand_keypoints_2d']
        left_hand_keypoints_2d[:, 0] -= left_hand_bbox[0]
        left_hand_keypoints_2d[:, 1] -= left_hand_bbox[1]
        results['left_hand_keypoints_2d'] = left_hand_keypoints_2d

        right_hand_keypoints_2d = results['right_hand_keypoints_2d']
        right_hand_keypoints_2d[:, 0] -= right_hand_bbox[0]
        right_hand_keypoints_2d[:, 1] -= right_hand_bbox[1]
        results['right_hand_keypoints_2d'] = right_hand_keypoints_2d

        left_hand_img = self._crop_hand(img, left_hand_bbox, self.padding_size)
        right_hand_img = self._crop_hand(img, right_hand_bbox, self.padding_size)
        results['left_hand_img'] = left_hand_img
        results['right_hand_img'] = right_hand_img
        return results





@PIPELINES.register_module()
class ResizeImageWithName:
    def __init__(self, img_h, img_w, img_name, keypoints_name_list, interpolation=cv2.INTER_LINEAR):
        super().__init__()
        self.target_img_h = img_h
        self.target_img_w = img_w
        self.interpolation = interpolation
        self.img_name = img_name
        if type(keypoints_name_list) is not list:
            keypoints_name_list = [keypoints_name_list]
        self.keypoints_name_list = keypoints_name_list

    def __call__(self, results: dict) -> dict:
        img = results[self.img_name]
        img_h, img_w, c = img.shape
        img = cv2.resize(img, dsize=(self.target_img_w, self.target_img_h), interpolation=cv2.INTER_LINEAR)
        results[self.img_name] = img

        scale_w = self.target_img_w / img_w
        scale_h = self.target_img_h / img_h

        # the 2d joints should also be treated here
        for keypoints_name in self.keypoints_name_list:
            if keypoints_name in results:
                joints_2d = results[keypoints_name]
                joints_2d[:, 0] *= scale_w
                joints_2d[:, 1] *= scale_h
                results[keypoints_name] = joints_2d
        return results


@PIPELINES.register_module()
class RGB2BGRHand:
    def __init__(self):
        super().__init__()

    def __call__(self, results: dict) -> dict:
        left_hand_img = results['left_hand_img']
        left_hand_img = cv2.cvtColor(left_hand_img, cv2.COLOR_RGB2BGR)
        results['left_hand_img'] = left_hand_img

        right_hand_img = results['right_hand_img']
        right_hand_img = cv2.cvtColor(right_hand_img, cv2.COLOR_RGB2BGR)
        results['right_hand_img'] = right_hand_img
        return results

@PIPELINES.register_module()
class ToTensorHand:
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.to_tensor = transforms.ToTensor()

    def __call__(self, results: dict) -> dict:
        left_hand_img = results['left_hand_img']
        left_hand_img = self.to_tensor(left_hand_img.astype(np.float32) / 255.0).to(self.device)
        results['left_hand_img'] = left_hand_img

        right_hand_img = results['right_hand_img']
        right_hand_img = self.to_tensor(right_hand_img.astype(np.float32) / 255.0).to(self.device)
        results['right_hand_img'] = right_hand_img
        return results
