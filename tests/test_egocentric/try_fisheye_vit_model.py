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
    fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
    data_cfg = dict(
        num_joints=15,
        camera_param_path=fisheye_camera_path,
        joint_type='mo2cap2',
        image_size=[img_res, img_res],
        heatmap_size=(64, 64),
        joint_weights=[1.] * 15,
        use_different_joint_weights=False,
    )

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='CropCircle', img_h=1024, img_w=1280),
        dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
        dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
        dict(type='ResizeImage', img_h=img_res, img_w=img_res),
        dict(type='Generate2DPoseConfidence'),
        dict(type='ToTensor'),
        dict(
            type='NormalizeTensor',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        dict(
            type='Collect',
            keys=[
                'img', 'keypoints_3d',
            ],
            meta_keys=['image_file', 'joints_2d']),
    ]

    train = dict(
        type='RenderpeopleMixamoDataset',
        ann_file='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo/renderpeople_mixamo_labels.pkl',
        img_prefix='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        test_mode=False,
    )

    custom_dataset = build_dataset(train)

    print(f'length of dataset is: {len(custom_dataset)}')

    data_i = custom_dataset[image_id]

    image_i = data_i['img']
    # cv2.imshow('img', image_i[:, :, ::-1])
    # cv2.waitKey(0)
    return image_i


def try_fisheye_vit():
    image_i = build_mocap_studio_dataset(image_id=1785)
    image_i = torch.asarray(image_i).unsqueeze(0).float()
    image_i = torch.repeat_interleave(image_i, repeats=4, dim=0)
    print(image_i.shape)
    fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
    # model = dict(
    #     type='Egocentric3DPoseEstimator',
    #     backbone=dict(
    #         type='FisheyeViT',
    #         img_size=(256, 256),
    #         patch_size=32,
    #         embed_dim=256,
    #         depth=6,
    #         num_heads=16,
    #         use_checkpoint=False,
    #         mlp_ratio=4,
    #         qkv_bias=True,
    #         drop_path_rate=0.3,
    #         fisheye2sphere_configs=dict(
    #             type='Fisheye2Sphere',
    #             input_feature_height=256,
    #             input_feature_width=256,
    #             image_h=1024,
    #             image_w=1280,
    #             patch_num_lat=10,
    #             patch_num_lon=20,
    #             patch_size=(0.2, 0.2),
    #             patch_pixel_number=(32, 32),
    #             crop_to_square=True,
    #             camera_param_path=fisheye_camera_path,
    #         )
    #     ),
    #     keypoint_head=dict(
    #         type='Regress3DPoseSimpleHead',
    #         in_channels=256,
    #         out_joint_num=15,
    #         num_conv_layers=0,
    #         num_fc_layers=2,
    #         num_fc_features=(46080, 1024, 45),
    #         loss_keypoint=dict(type='MPJPELoss', use_target_weight=False)),
    #     train_cfg=dict(),
    #     test_cfg=dict()
    # )

    model = dict(
        type='Egocentric3DPoseEstimator',
        backbone=dict(
            type='ViTJointToken',
            img_size=(256, 256),
            patch_size=16,
            embed_dim=768,
            num_keypoints=15,
            depth=6,
            num_heads=12,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.3,
        ),
        keypoint_head=dict(
            type='Regress3DPoseSimpleHead',
            in_channels=768,
            out_joint_num=15,
            num_conv_layers=0,
            num_fc_layers=2,
            num_fc_features=(768 * 15, 768 * 4, 3 * 15),
            loss_keypoint=dict(type='MPJPELoss', use_target_weight=False)),
        train_cfg=dict(),
        test_cfg=dict())

    # build egocentric 3d pose pipeline
    network = build_posenet(model)
    res = network(image_i)
    print(res.shape)


if __name__ == '__main__':
    try_fisheye_vit()
