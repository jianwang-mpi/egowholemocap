import torch
import numpy as np
import cv2
from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import (LoadImageFromFile, CropCircle, Generate2DPose,
                                       CropImage, ResizeImage)

from mmpose.models import build_posenet


def build_mocap_studio_dataset(image_id=785):
    dataset_name = 'MocapStudioDataset'
    print(f'test dataset: {dataset_name}')
    img_res = 256
    fisheye_camera_path = 'Z:/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
    data_cfg = dict(
        num_joints=15,
        camera_param_path=fisheye_camera_path,
        joint_type='mo2cap2',
        image_size=[img_res, img_res],
        heatmap_size=(64, 64),
        joint_weights=[1.] * 15,
        use_different_joint_weights=False,
    )

    pipeline_vis_image = [
        LoadImageFromFile(),
        CropCircle(img_h=1024, img_w=1280),
        Generate2DPose(fisheye_model_path=fisheye_camera_path),
        CropImage(crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
        ResizeImage(img_h=img_res, img_w=img_res)
    ]

    dataset_cfg = dict(
        type=dataset_name,
        data_cfg=data_cfg,
        pipeline=pipeline_vis_image,
        local=True,
        test_mode=True, )

    custom_dataset = build_dataset(dataset_cfg)

    assert custom_dataset.test_mode is True
    print(f'length of dataset is: {len(custom_dataset)}')

    data_i = custom_dataset[image_id]

    image_i = data_i['img']
    # cv2.imshow('img', image_i[:, :, ::-1])
    # cv2.waitKey(0)
    return image_i


def try_fisheye_to_sphere():
    image_i = build_mocap_studio_dataset(image_id=1785)
    image_i = torch.asarray(image_i).unsqueeze(0).float()
    image_i = torch.permute(image_i, (0, 3, 1, 2))
    # build the fisheye to spere model
    module_name = 'Fisheye2Sphere'

    fisheye2sphere_config = dict(
        type=module_name,
        input_feature_height=256,
        input_feature_width=256,
        image_h=1024,
        image_w=1280,
        patch_num_lat=16,
        patch_num_lon=16,
        patch_size=(0.1, 0.1),
        patch_pixel_number=(64, 64),
        crop_to_square=True,
        camera_param_path='Z:/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json',
        )

    fisheye2sphere = build_posenet(fisheye2sphere_config)
    image_patches = fisheye2sphere(image_i)
    image_patches = image_patches[0]
    print(image_patches.shape)

    image_patches = image_patches.numpy().astype(np.uint8)

    for i, patch_i in enumerate(image_patches):
        patch_i = np.transpose(patch_i, axes=(1, 2, 0))
        patch_i = patch_i[:, :, ::-1]
        cv2.imwrite('Z:/EgoMocap/work/EgocentricFullBody/vis_results/patch_moving_image/img_%03d.jpg' % i, patch_i)


if __name__ == '__main__':
    # test_ego_2d_dataset()
    try_fisheye_to_sphere()
