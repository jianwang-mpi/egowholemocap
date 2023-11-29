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
from mmpose.utils.visualization.draw import draw_joints
from mmpose.utils.visualization.skeleton import Skeleton


def load_image(image_id = 785):
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

    # dataset_cfg = dict(
    #     type='RenderpeopleMixamoDataset',
    #     ann_file='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo/renderpeople_mixamo_labels_old.pkl',
    #     img_prefix='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo',
    #     data_cfg=data_cfg,
    #     pipeline=pipeline,
    #     test_mode=False,
    # )

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
        type='RefineNetMLP',
        fisheye_calibration_file=fisheye_camera_path,
        add_joint_noise=False,
        noise_std=0.002
    )

    # build egocentric 3d pose pipeline
    network = build_posenet(model)

    # load image
    data_i = load_image(image_id=13700)
    forward_results = network(data_i)

    print(forward_results['keypoints_pred'].shape)

    # visualize 3d joints
    joints_3d = forward_results['keypoints_3d_w_noise'][1].cpu().numpy()
    joints_3d_wo_noise = forward_results['keypoints_3d_wo_noise'][1].detach().cpu().numpy()

    skeleton = Skeleton(None)
    mesh_3d = skeleton.joints_2_mesh(joints_3d)
    mesh_3d_wo_noise = skeleton.joints_2_mesh(joints_3d_wo_noise)

    open3d.io.write_triangle_mesh('/CT/EgoMocap/work/EgocentricFullBody/vis_results/crop_img/joints_w_noise.ply',
                                  mesh_3d)
    open3d.io.write_triangle_mesh('/CT/EgoMocap/work/EgocentricFullBody/vis_results/crop_img/joints_wo_noise.ply',
                                    mesh_3d_wo_noise)


    cropped_image = network.debug_crop_img(data_i)
    cropped_image = cropped_image.view(-1, 15,  3, 256, 256)[1]

    print(cropped_image.shape)
    cropped_image = cropped_image.cpu().numpy()
    # visualize cropped image
    for i in range(15):
        img_part = cropped_image[i]
        img_part *= np.asarray([0.229, 0.224, 0.225])[:, None, None]
        img_part += np.asarray([0.485, 0.456, 0.406])[:, None, None]

        img_part = np.transpose(img_part, (1, 2, 0)) * 255
        img_part = img_part.astype(np.uint8)
        img_part = cv2.cvtColor(img_part, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'/CT/EgoMocap/work/EgocentricFullBody/vis_results/crop_img/test_{i}.png', img_part)

    # visualize the joints
    fisheye_camera = FishEyeCameraCalibrated(fisheye_camera_path)
    joints_3d = data_i['keypoints_3d'][0].cpu().numpy()
    joints_2d = fisheye_camera.world2camera(joints_3d)

    original_image = data_i['original_image'].cpu().numpy()[0]
    original_image *= np.asarray([0.229, 0.224, 0.225])[:, None, None]
    original_image += np.asarray([0.485, 0.456, 0.406])[:, None, None]

    original_image = np.transpose(original_image, (1, 2, 0)) * 255
    original_image = original_image.astype(np.uint8)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    img_with_joints = draw_joints(joints_2d, original_image)
    cv2.imwrite(f'/CT/EgoMocap/work/EgocentricFullBody/vis_results/crop_img/img_with_joints.png', img_with_joints)



if __name__ == '__main__':
    try_refine_net()
