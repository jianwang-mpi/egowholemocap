import torch
import numpy as np
import cv2
from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import (LoadImageFromFile, CropCircle, Generate2DPose,
                                       CropImage, ResizeImage)

from mmpose.models import build_posenet


def build_mocap_studio_dataset(image_id=785):
    img_res = 256
    fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
    data_cfg = dict(
        camera_param_path=fisheye_camera_path,
        joint_type='smplx',
        image_size=[img_res, img_res],
    )

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='CropCircle', img_h=1024, img_w=1280),
        dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
        dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
        dict(type='ResizeImage', img_h=img_res, img_w=img_res),
        dict(type='ToTensor'),
        dict(
            type='NormalizeTensor',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        dict(
            type='Collect',
            keys=[
                'img', 'keypoints_2d', 'keypoints_2d_visible', 'keypoints_3d', 'keypoints_3d_visible'
            ],
            meta_keys=[]),
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
    return data_i


def try_fisheye_vit():
    image_i = build_mocap_studio_dataset(image_id=1785)
    image_i = torch.asarray(image_i).unsqueeze(0).float()
    image_i = torch.repeat_interleave(image_i, repeats=4, dim=0)
    print(image_i.shape)
    fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
    model = dict(
        type='SMPLXNetWrapper',
        exp_cfg_path='/CT/EgoMocap/work/EgocentricFullBody/configs/body/egofullbody/renderpeople_mixamo/expose/expose_conf_only_body.yaml',
        smplx_loss=dict(
            type='FisheyeMeshLoss',
            joints_2d_loss_weight=1e-4,
            joints_3d_loss_weight=5,
            vertex_loss_weigh=0,
            smpl_pose_loss_weight=0,
            smpl_beta_loss_weight=0,
            camera_param_path=fisheye_camera_path
        ),
        pretrained='/CT/EgoMocap/work/EgocentricFullBody/resources/pretrained_models/expose/checkpoints'
    )

    # build egocentric 3d pose pipeline
    network = build_posenet(model)
    res = network(image_i)
    print(res.shape)


if __name__ == '__main__':
    try_fisheye_vit()
