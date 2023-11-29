import cv2
import numpy as np
import open3d

from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import (LoadImageFromFile, CropCircle, Generate2DPose,
                                       CropImage, ResizeImage, Generate2DPoseConfidence, ToTensor, NormalizeTensor,
                                       TopDownGenerateTarget, Collect)
from mmpose.utils.visualization.draw import draw_joints
from mmpose.utils.visualization.skeleton import Skeleton


def test_ego_2d_dataset():
    dataset_name = 'RenderpeopleMixamoDataset'
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

    pipeline_vis_image = [
        LoadImageFromFile(),
        CropCircle(img_h=1024, img_w=1280),
        Generate2DPose(fisheye_model_path=fisheye_camera_path),
        CropImage(crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
        ResizeImage(img_h=img_res, img_w=img_res)
    ]
    pipeline_show_pose = [
        LoadImageFromFile(),
        CropCircle(img_h=1024, img_w=1280),
        Generate2DPose(fisheye_model_path=fisheye_camera_path),
        CropImage(crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
        ResizeImage(img_h=img_res, img_w=img_res),
        Generate2DPoseConfidence(),
        ToTensor(),
        NormalizeTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        TopDownGenerateTarget(sigma=1.5),
        Collect(keys=['img', 'target', 'target_weight'],
                meta_keys=['image_file', 'keypoints_3d', 'joints_3d', 'joints_3d_visible', 'joints_2d'])
    ]

    dataset_cfg = dict(
        type=dataset_name,
        ann_file='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo/renderpeople_mixamo_labels_old.pkl',
        img_prefix='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo',
        data_cfg=data_cfg,
        pipeline=pipeline_show_pose,
        test_mode=False)

    custom_dataset = build_dataset(dataset_cfg)

    assert custom_dataset.test_mode is False
    print(f'length of dataset is: {len(custom_dataset)}')

    image_id = 785
    data_i = custom_dataset[image_id]

    image_i = data_i['img']
    # convert to bgr and save with opencv
    image_i = image_i.cpu().numpy()
    for i in range(3):

        image_i[i] *= np.asarray([0.229, 0.224, 0.225])[i]
        image_i[i] += np.asarray([0.485, 0.456, 0.406])[i]
    image_i = np.transpose(image_i, (1, 2, 0))
    image_i_bgr = image_i[:, :, ::-1]
    image_i_bgr = (image_i_bgr * 255).astype(np.uint8)

    cv2.imwrite('/CT/EgoMocap/work/EgocentricFullBody/vis_results/out_recovered.jpg', image_i_bgr)

    # joint 2d visualize
    print(data_i['img_metas'].data['image_file'])
    joint_2d = data_i['img_metas'].data['joints_2d']
    image_i_bgr = draw_joints(joint_2d, image_i_bgr)
    cv2.imwrite('/CT/EgoMocap/work/EgocentricFullBody/vis_results/joints_2d.jpg', image_i_bgr)

    # visualize the heatmap
    heatmap_2d = data_i['target']
    heatmap_2d = np.sum(heatmap_2d, axis=0)
    heatmap_2d /= np.max(heatmap_2d)
    heatmap_2d = (heatmap_2d * 255).astype(np.uint8)
    cv2.imwrite('/CT/EgoMocap/work/EgocentricFullBody/vis_results/heatmap.jpg', heatmap_2d)

    # visualize the 3d pose
    keypoints_3d = data_i['img_metas'].data['keypoints_3d']
    skeleton = Skeleton(None)
    skeleton_mesh = skeleton.joints_2_mesh(keypoints_3d)
    coor = open3d.geometry.TriangleMesh.create_coordinate_frame()
    open3d.io.write_triangle_mesh('/CT/EgoMocap/work/EgocentricFullBody/vis_results/mesh.ply', skeleton_mesh + coor)

    # visualize the projected 2d pose


def test_mocap_studio_dataset():
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

    pipeline_show_pose = [
        LoadImageFromFile(),
        CropCircle(img_h=1024, img_w=1280),
        Generate2DPose(fisheye_model_path=fisheye_camera_path),
        CropImage(crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
        ResizeImage(img_h=img_res, img_w=img_res),
        Generate2DPoseConfidence(),
        ToTensor(),
        NormalizeTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        TopDownGenerateTarget(sigma=1.5),
        dict(
            type='Collect',
            keys=[
                'img',
            ],
            meta_keys=['image_file', 'target', 'joints_2d']),
    ]

    dataset_cfg = dict(
        type=dataset_name,
        data_cfg=data_cfg,
        pipeline=pipeline_show_pose,
        test_mode=True,)

    custom_dataset = build_dataset(dataset_cfg)

    assert custom_dataset.test_mode is True
    print(f'length of dataset is: {len(custom_dataset)}')

    image_id = 785
    data_i = custom_dataset[image_id]

    image_i = data_i['img']
    # convert to bgr and save with opencv
    image_i = image_i.cpu().numpy()
    for i in range(3):
        image_i[i] *= np.asarray([0.229, 0.224, 0.225])[i]
        image_i[i] += np.asarray([0.485, 0.456, 0.406])[i]
    image_i = np.transpose(image_i, (1, 2, 0))
    image_i_bgr = image_i[:, :, ::-1]
    image_i_bgr = (image_i_bgr * 255).astype(np.uint8)

    cv2.imwrite('/CT/EgoMocap/work/EgocentricFullBody/vis_results/out_recovered.jpg', image_i_bgr)

    # joint 2d visualize
    print(data_i['img_metas'].data['image_file'])
    joint_2d = data_i['img_metas'].data['joints_2d']
    image_i_bgr = draw_joints(joint_2d, image_i_bgr)
    cv2.imwrite('/CT/EgoMocap/work/EgocentricFullBody/vis_results/joints_2d.jpg', image_i_bgr)

    # visualize the heatmap
    heatmap_2d = data_i['img_metas'].data['target']
    heatmap_2d = np.sum(heatmap_2d, axis=0)
    heatmap_2d /= np.max(heatmap_2d)
    heatmap_2d = (heatmap_2d * 255).astype(np.uint8)
    cv2.imwrite('/CT/EgoMocap/work/EgocentricFullBody/vis_results/heatmap.jpg', heatmap_2d)



def test_mocap_studio_finetune_dataset():
    dataset_name = 'MocapStudioFinetuneDataset'
    print(f'test dataset: {dataset_name}')
    img_res = 256
    fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
    # fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration.json'

    skeleton = Skeleton(None)
    data_cfg = dict(
        num_joints=15,
        camera_param_path=fisheye_camera_path,
        joint_type='mo2cap2',
        image_size=[img_res, img_res],
        heatmap_size=(64, 64),
        joint_weights=[1.] * 15,
        use_different_joint_weights=False,
    )

    pipeline_show_pose = [
        LoadImageFromFile(),
        CropCircle(img_h=1024, img_w=1280),
        Generate2DPose(fisheye_model_path=fisheye_camera_path),
        CropImage(crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
        ResizeImage(img_h=img_res, img_w=img_res),
        Generate2DPoseConfidence(),
        ToTensor(),
        NormalizeTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        TopDownGenerateTarget(sigma=1.5),
        dict(
            type='Collect',
            keys=[
                'img', 'keypoints_3d'
            ],
            meta_keys=['image_file', 'target', 'keypoints_2d']),
    ]

    dataset_cfg = dict(
        type=dataset_name,
        data_cfg=data_cfg,
        pipeline=pipeline_show_pose,
        test_mode=True,)

    custom_dataset = build_dataset(dataset_cfg)

    assert custom_dataset.test_mode is True
    print(f'length of dataset is: {len(custom_dataset)}')

    image_id = 3000
    data_i = custom_dataset[image_id]

    image_i = data_i['img']
    # convert to bgr and save with opencv
    image_i = image_i.cpu().numpy()
    for i in range(3):
        image_i[i] *= np.asarray([0.229, 0.224, 0.225])[i]
        image_i[i] += np.asarray([0.485, 0.456, 0.406])[i]
    image_i = np.transpose(image_i, (1, 2, 0))
    image_i_bgr = image_i[:, :, ::-1]
    image_i_bgr = (image_i_bgr * 255).astype(np.uint8)

    cv2.imwrite('/CT/EgoMocap/work/EgocentricFullBody/vis_results/out_recovered.jpg', image_i_bgr)

    # joint 2d visualize
    print(data_i['img_metas'].data['image_file'])
    joint_2d = data_i['img_metas'].data['keypoints_2d']
    image_i_bgr = draw_joints(joint_2d, image_i_bgr)
    cv2.imwrite('/CT/EgoMocap/work/EgocentricFullBody/vis_results/joints_2d.jpg', image_i_bgr)

    # visualize the heatmap
    heatmap_2d = data_i['img_metas'].data['target']
    heatmap_2d = np.sum(heatmap_2d, axis=0)
    heatmap_2d /= np.max(heatmap_2d)
    heatmap_2d = (heatmap_2d * 255).astype(np.uint8)
    cv2.imwrite('/CT/EgoMocap/work/EgocentricFullBody/vis_results/heatmap.jpg', heatmap_2d)

    # visualize the 3d pose
    pose_3d = data_i['keypoints_3d']
    skeleton_mesh = skeleton.joints_2_mesh(pose_3d)
    coor = open3d.geometry.TriangleMesh.create_coordinate_frame()
    open3d.io.write_triangle_mesh('/CT/EgoMocap/work/EgocentricFullBody/vis_results/mesh.ply', skeleton_mesh + coor)

if __name__ == '__main__':
    # test_ego_2d_dataset()
    # test_mocap_studio_dataset()
    test_mocap_studio_finetune_dataset()