import cv2
import numpy as np
from copy import copy, deepcopy
from mmpose.datasets.builder import PIPELINES
from mmpose.utils.visualization.skeleton import Skeleton
import math
import torch
from torchvision.transforms import functional as F


@PIPELINES.register_module()
class CopyImage:
    def __init__(self, source='img', target='img_copy'):
        super().__init__()
        self.source = source
        self.target = target

    def __call__(self, results: dict) -> dict:
        results[self.target] = deepcopy(results[self.source])
        return results

@PIPELINES.register_module()
class ToTensorWithName:

    def __init__(self, device='cpu', img_name='img'):
        self.device = device
        self.img_name = img_name

    def _to_tensor(self, x):
        return torch.from_numpy(x.astype('float32')).permute(2, 0, 1).to(
            self.device).div_(255.0)

    def __call__(self, results):
        results[self.img_name] = self._to_tensor(results[self.img_name])

        return results


@PIPELINES.register_module()
class NormalizeTensorWithName:

    def __init__(self, mean, std, img_name='img'):
        self.mean = mean
        self.std = std

        self.img_name = img_name

    def __call__(self, results):
        if isinstance(results[self.img_name], (list, tuple)):
            results[self.img_name] = [
                F.normalize(img, mean=self.mean, std=self.std, inplace=True)
                for img in results[self.img_name]
            ]
        else:
            results[self.img_name] = F.normalize(
                results[self.img_name], mean=self.mean, std=self.std, inplace=True)

        return results


@PIPELINES.register_module()
class Generate2DPose:
    def __init__(self, fisheye_model_path):
        super().__init__()
        self.skeleton_model = Skeleton(fisheye_model_path)

    def __call__(self, results: dict) -> dict:
        pose_3d = results['keypoints_3d']
        results['keypoints_2d_visible'] = results['keypoints_3d_visible']
        # solved: some joints are: (0, 0, 0)
        pose_2d = np.zeros((pose_3d.shape[0], 2))
        pose_3d_vis = pose_3d[results['keypoints_3d_visible'] > 0]
        pose_2d_vis = self.skeleton_model.camera.world2camera(pose_3d_vis)
        pose_2d[results['keypoints_2d_visible'] > 0] = pose_2d_vis

        results['keypoints_2d'] = pose_2d

        return results


@PIPELINES.register_module()
class Generate2DPoseConfidence:
    def __init__(self):
        super().__init__()

    def __call__(self, results: dict) -> dict:
        joints_2d = results['keypoints_2d']
        results['joints_2d'] = joints_2d
        joints_3d = np.zeros((joints_2d.shape[0], 3))
        joints_3d[:, :2] = joints_2d
        joints_3d[:, 2] = deepcopy(results['keypoints_2d_visible'])
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = np.expand_dims(results['keypoints_2d_visible'], axis=1)
        return results


@PIPELINES.register_module()
class CropImage:
    def __init__(self, crop_left, crop_right, crop_top, crop_bottom):
        super().__init__()
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

    def __call__(self, results: dict) -> dict:
        img = results['img']
        img_h, img_w, c = img.shape
        img = img[self.crop_top: img_h-self.crop_bottom, self.crop_left: img_w-self.crop_right, :]
        results['img'] = img

        # the 2d joints should also be treated here
        if 'keypoints_2d' in results:
            joint_2d = results['keypoints_2d']
            joint_2d[:, 0] -= self.crop_left
            joint_2d[:, 1] -= self.crop_top
            results['keypoints_2d'] = joint_2d
        return results


@PIPELINES.register_module()
class ResizeImage:
    def __init__(self, img_h, img_w, interpolation=cv2.INTER_LINEAR):
        super().__init__()
        self.target_img_h = img_h
        self.target_img_w = img_w
        self.interpolation = interpolation

    def __call__(self, results: dict) -> dict:
        img = results['img']
        img_h, img_w, c = img.shape
        img = cv2.resize(img, dsize=(self.target_img_w, self.target_img_h), interpolation=cv2.INTER_LINEAR)
        results['img'] = img

        scale_w = self.target_img_w / img_w
        scale_h = self.target_img_h / img_h

        if 'keypoints_2d' in results:
            joints_2d = results['keypoints_2d']
            joints_2d[:, 0] *= scale_w
            joints_2d[:, 1] *= scale_h
            results['keypoints_2d'] = joints_2d
        return results


@PIPELINES.register_module()
class CropCircle:
    def __init__(self, img_h, img_w, center=None, radius=None):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.mask = self.make_circle_mask(img_h, img_w, center, radius)

    def make_circle_mask(self, img_h=1024, img_w=1280, center=None, radius=None):
        circle_mask = np.zeros(shape=(img_h, img_w, 3), dtype=np.uint8)
        if center is None:
            center = (img_w // 2, img_h // 2)
        if radius is None:
            radius = int(360 / 1024 * img_h * np.sqrt(2))
        circle_mask = cv2.circle(circle_mask, center=center, radius=radius,
                                 color=(255, 255, 255), thickness=-1)
        circle_mask = (circle_mask > 0).astype(np.uint8)
        return circle_mask

    def __call__(self, results: dict) -> dict:
        img = results['img']

        # resize if the image size is different
        img_h, img_w, c = img.shape
        if img_h != self.img_h or img_w != self.img_w:
            scale_h = img_h / self.img_h
            scale_w = img_w / self.img_w
            if math.isclose(scale_w, scale_h, rel_tol=1e-5):
                img = cv2.resize(img, dsize=(self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
            else:
                raise Exception('The ratio of height and width of input image does not match')

        results['img'] = self.mask * img
        return results
