#  Copyright Jian Wang @ MPI-INF (c) 2023.

import numpy as np
import open3d
import torch

from mmpose.datasets.builder import PIPELINES
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated


@PIPELINES.register_module()
class CropHandImageFisheye:
    def __init__(self, fisheye_camera_path, input_img_h, input_img_w,
                 crop_img_size=256, enlarge_scale=1.2):
        super(CropHandImageFisheye, self).__init__()
        self.enlarge_scale = enlarge_scale
        self.fisheye_camera = FishEyeCameraCalibrated(fisheye_camera_path)
        # self.crop_to_square = crop_to_square
        self.crop_img_size = crop_img_size

        self.input_img_h = input_img_h
        self.input_img_w = input_img_w

    def _project_patches_to_original_image(self, patch):
        H, W, _ = patch.shape
        patch_flat = patch.reshape((-1, 3))
        pos_2d_flat = self.fisheye_camera.world2camera(patch_flat)
        pose_2d_list = pos_2d_flat.reshape((H, W, 2))
        # in patch pixel position (H, W), the first dimension of pose_2d is x position, second dimension is y position
        # if self.crop_to_square:
        #     image_w_crop_left = (self.image_w - self.image_h) // 2
        #     # image_w_crop_right = (self.image_w - self.image_h) // 2
        #     pose_2d_list[:, :, 0] -= image_w_crop_left
        # resize to [-1, 1]
        pose_2d_list[:, :, 0] = (pose_2d_list[:, :, 0] - self.input_img_w / 2) / (self.input_img_w / 2)
        pose_2d_list[:, :, 1] = (pose_2d_list[:, :, 1] - self.input_img_h / 2) / (self.input_img_h / 2)

        pose_2d_list = np.asarray(pose_2d_list)  # shape: (H, W, 2)
        return pose_2d_list

    def _generate_patch_coordinates(self, patch_center_3d, patch_x_end_ray, patch_pixel_num, visualize=False):
        tangent_plane_normal = patch_center_3d
        crossing_point = ((patch_center_3d @ tangent_plane_normal) / (
                patch_x_end_ray @ tangent_plane_normal)) * patch_x_end_ray
        up_vector = crossing_point - patch_center_3d

        # generate (x, y) pairs
        x_range = np.arange(0, patch_pixel_num[0]) / (patch_pixel_num[0] - 1) - 0.5
        y_range = np.arange(0, patch_pixel_num[1]) / (patch_pixel_num[1] - 1) - 0.5

        x_range *= 2
        y_range *= 2

        x_range *= np.linalg.norm(up_vector)
        y_range *= np.linalg.norm(up_vector)

        right_vector = np.cross(tangent_plane_normal, up_vector)
        right_vector /= np.linalg.norm(right_vector)
        down_vector = -up_vector  # use down vector to match the opencv traditional image coordinate system
        down_vector /= np.linalg.norm(down_vector)

        x_axis = right_vector.copy()
        y_axis = down_vector.copy()
        z_axis = (patch_center_3d / np.linalg.norm(patch_center_3d)).copy()
        patch_coordinate_system_rot = np.vstack((x_axis, y_axis, z_axis)).T  # rotation matrix multiply on the left
        patch_coordinate_system_transl = patch_center_3d.copy()
        patch_coordinate_system_transform = np.eye(4)
        patch_coordinate_system_transform[:3, :3] = patch_coordinate_system_rot
        patch_coordinate_system_transform[:3, 3] = patch_coordinate_system_transl

        patch_centers = np.dstack(np.meshgrid(x_range, y_range)).reshape(-1, 2)
        patch_coordinates = patch_centers[:, 0].reshape(-1, 1) * right_vector + patch_centers[:, 1].reshape(-1,
                                                                                                            1) * down_vector + patch_center_3d

        if visualize:
            sphere_list = []
            coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
            local_coord = coord.transform(patch_coordinate_system_transform)
            for patch_center in patch_coordinates:
                sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(patch_center)
                sphere_list.append(sphere)
            open3d.visualization.draw_geometries(sphere_list + [coord, local_coord])

        patch_coordinates = patch_coordinates.reshape((len(x_range), len(y_range), 3))
        return patch_coordinates, patch_coordinate_system_transform

    def _crop_hand_img(self, img, patch_coordinates):
        H, W, C = img.shape
        assert C == 3 and H == self.input_img_h and W == self.input_img_w
        H_patch, W_patch, _ = patch_coordinates.shape
        assert H_patch == W_patch
        assert _ == 2
        img_torch = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        patch_coordinates_torch = torch.from_numpy(patch_coordinates).float().unsqueeze(0)
        sampled_patch = torch.nn.functional.grid_sample(img_torch, patch_coordinates_torch, align_corners=True)
        sampled_patch = sampled_patch.view(C, H_patch, W_patch).contiguous()
        sampled_patch = sampled_patch.permute(1, 2, 0).numpy()
        return sampled_patch

    def _create_hand_bbox(self, hand_keypoints_2d, enlarge_scale):
        hand_bbox = np.zeros(4)
        hand_bbox[0: 2] = np.min(hand_keypoints_2d, axis=0)
        hand_bbox[2: 4] = np.max(hand_keypoints_2d, axis=0)

        bbox_w = hand_bbox[2] - hand_bbox[0]
        bbox_h = hand_bbox[3] - hand_bbox[1]
        bbox_center = np.array([hand_bbox[0] + bbox_w / 2, hand_bbox[1] + bbox_h / 2])
        bbox_size = max(bbox_w, bbox_h) * enlarge_scale

        return bbox_center, bbox_size

    def _crop_hand(self, img, hand_center, hand_crop_size):
        hand_up = np.array([hand_center[0], hand_center[1] - hand_crop_size / 2])
        hand_center = np.reshape(hand_center, (1, 2))
        hand_up = np.reshape(hand_up, (1, 2))
        hand_center_3d = self.fisheye_camera.camera2world(hand_center, np.ones((1,)))
        hand_up_3d = self.fisheye_camera.camera2world(hand_up, np.ones((1,)))

        hand_center_3d = hand_center_3d.reshape(3)
        hand_up_3d = hand_up_3d.reshape(3)

        hand_patch_3d, hand_patch_transform = self._generate_patch_coordinates(hand_center_3d, hand_up_3d,
                                                                               (self.crop_img_size, self.crop_img_size),
                                                                               visualize=False)
        hand_patch_2d = self._project_patches_to_original_image(hand_patch_3d)
        hand_patch_img = self._crop_hand_img(img, hand_patch_2d)
        return hand_patch_img, hand_patch_transform, hand_patch_2d

    def __call__(self, results: dict) -> dict:
        img = results['img']
        img_h, img_w, _ = img.shape
        left_hand_keypoints_2d = results['left_hand_keypoints_2d']
        right_hand_keypoints_2d = results['right_hand_keypoints_2d']
        left_hand_center, left_hand_crop_size = self._create_hand_bbox(left_hand_keypoints_2d, self.enlarge_scale)
        right_hand_center, right_hand_crop_size = self._create_hand_bbox(right_hand_keypoints_2d, self.enlarge_scale)

        results['left_hand_center'] = left_hand_center
        results['right_hand_center'] = right_hand_center
        results['left_hand_crop_size'] = left_hand_crop_size
        results['right_hand_crop_size'] = right_hand_crop_size

        left_hand_patch_img, left_hand_transform, left_hand_patch_2d = self._crop_hand(img, left_hand_center, left_hand_crop_size)
        right_hand_patch_img, right_hand_transform, right_hand_patch_2d = self._crop_hand(img, right_hand_center, right_hand_crop_size)

        # todo: the 2d and 3d joints should also be treated here

        results['left_hand_img'] = left_hand_patch_img
        results['left_hand_transform'] = left_hand_transform
        results['left_hand_patch_2d'] = left_hand_patch_2d
        results['right_hand_img'] = right_hand_patch_img
        results['right_hand_transform'] = right_hand_transform
        results['right_hand_patch_2d'] = right_hand_patch_2d
        return results
