import open3d
import torch
import numpy as np
import cv2
from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import (LoadImageFromFile, CropCircle, Generate2DPose,
                                       CropImage, ResizeImage, Generate2DPoseConfidence, ToTensor, NormalizeTensor,
                                       TopDownGenerateTarget, Collect)
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated

from mmpose.models import build_posenet
from mmpose.utils.visualization.draw import draw_joints, draw_bbox
from mmpose.utils.visualization.skeleton import Skeleton


def load_data(image_id=785):
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

    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='CropCircle', img_h=1024, img_w=1280),
        dict(type='CopyImage', source='img', target='original_image'),
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
            type='ToTensorWithName',
            img_name='original_image'
        ),
        dict(
            type='NormalizeTensorWithName',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            img_name='original_image'
        ),
        dict(
            type='Collect',
            keys=[
                'img', 'original_image', 'keypoints_3d', 'keypoints_3d_visible',
            ],
            meta_keys=['image_file', 'joints_2d']),
    ]

    dataset_cfg = dict(
        type='MocapStudioFinetuneDataset',
        data_cfg=data_cfg,
        pipeline=pipeline,
        test_mode=False,
    )

    custom_dataset = build_dataset(dataset_cfg)

    assert custom_dataset.test_mode is False
    print(f'length of dataset is: {len(custom_dataset)}')

    data_i = custom_dataset[image_id]

    data_batch = {
        'keypoints_3d': torch.repeat_interleave(torch.unsqueeze(torch.from_numpy(data_i['keypoints_3d']), 0), 2, 0).contiguous(),
        'original_image': torch.repeat_interleave(torch.unsqueeze(data_i['original_image'], 0), 2, 0).contiguous(),
    }

    return data_batch

def try_refine_net():
    fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
    model = dict(
        type='HandDetection',
        det_config='/CT/EgoMocap/work/EgocentricFullBody/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py',
        det_checkpoint='https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth',
        fisheye_calibration_file=fisheye_camera_path,
        crop_size=(256, 256),
    )

    # build egocentric 3d pose pipeline
    network = build_posenet(model)

    # load image
    data_i = load_data(image_id=1700)
    forward_results = network(data_i)

    print(forward_results['left_hand_bbox'].shape)
    print(forward_results['right_hand_bbox'].shape)

    # visualize bbox
    original_image = data_i['original_image'].cpu().numpy()[0]
    original_image *= np.asarray([0.229, 0.224, 0.225])[:, None, None]
    original_image += np.asarray([0.485, 0.456, 0.406])[:, None, None]

    original_image = np.transpose(original_image, (1, 2, 0)) * 255
    original_image = original_image.astype(np.uint8)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    left_hand_bbox = forward_results['left_hand_bbox'].cpu().numpy()[0]
    right_hand_bbox = forward_results['right_hand_bbox'].cpu().numpy()[0]

    original_image = draw_bbox(original_image, left_hand_bbox)
    original_image = draw_bbox(original_image, right_hand_bbox)

    cv2.imwrite(f'/CT/EgoMocap/work/EgocentricFullBody/vis_results/crop_img/hand_bbox.png', original_image)




if __name__ == '__main__':
    try_refine_net()
