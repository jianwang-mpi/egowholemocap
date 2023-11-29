import os.path

import cv2
import numpy as np
from tqdm import tqdm

from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines.ego_hand_crop_transform import CropHandImageFisheye
from mmpose.models.builder import build_posenet
from mmpose.datasets.pipelines import (LoadImageFromFile, CropCircle, Generate2DPose,
                                       CropImage, ResizeImage, Generate2DPoseConfidence, ToTensor, NormalizeTensor,
                                       TopDownGenerateTarget, Collect, Generate2DHandPose, CropHandImage,
                                       ResizeImageWithName, RGB2BGRHand, ToTensorHand)
from mmpose.utils.visualization.draw import draw_keypoints


def test_studio_with_hand_dataset():
    path_dict = {
        'jian1': {
            'path': r'/HPS/ScanNet/work/egocentric_view/05082022/jian1',
        },
        'jian2': {
            'path': r'/HPS/ScanNet/work/egocentric_view/05082022/jian2',
        },
        'diogo1': {
            'path': r'/HPS/ScanNet/work/egocentric_view/05082022/diogo1',
        },
        'diogo2': {
            'path': r'/HPS/ScanNet/work/egocentric_view/05082022/diogo2',
        },
        'pranay2': {
            'path': r'/HPS/ScanNet/work/egocentric_view/05082022/pranay2',
        },
    }
    dataset_name = 'MocapStudioHandDataset'
    print(f'test dataset: {dataset_name}')
    img_res = 256
    fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
    data_cfg = dict(
        num_joints=69,
        camera_param_path=fisheye_camera_path,
        joint_type='studio',
        image_size=[img_res, img_res],
        hand_image_size=[img_res, img_res],
        heatmap_size=(64, 64),
        joint_weights=[1.] * 69,
        use_different_joint_weights=False,
    )

    pipeline_show_pose = [
        LoadImageFromFile(),
        CropCircle(img_h=1024, img_w=1280),
        Generate2DPose(fisheye_model_path=fisheye_camera_path),
        Generate2DHandPose(fisheye_model_path=fisheye_camera_path),
        CropHandImageFisheye(fisheye_camera_path, input_img_h=1024, input_img_w=1280,
                 crop_img_size=256, enlarge_scale=1.3),
        # ResizeImageWithName(img_h=img_res, img_w=img_res, img_name='left_hand_img',
        #                     keypoints_name_list=['left_hand_keypoints_2d']),
        # ResizeImageWithName(img_h=img_res, img_w=img_res, img_name='right_hand_img',
        #                     keypoints_name_list=['right_hand_keypoints_2d']),
        RGB2BGRHand(),
        ToTensorHand(),
        CropImage(crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
        ResizeImage(img_h=img_res, img_w=img_res),
        Generate2DPoseConfidence(),
        ToTensor(),
        NormalizeTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        TopDownGenerateTarget(sigma=1.5),
        Collect(keys=['img', 'target', 'target_weight'],
                meta_keys=['image_file', 'keypoints_3d', 'joints_3d', 'joints_3d_visible',
                           'left_hand_keypoints_3d', 'right_hand_keypoints_3d',
                           'left_hand_img', 'right_hand_img',
                           'left_hand_transform', 'right_hand_transform',
                           'left_hand_patch_2d', 'right_hand_patch_2d']),
    ]

    dataset_cfg = dict(
        type=dataset_name,
        path_dict=path_dict,  # none means use default path, test dataset
        data_cfg=data_cfg,
        pipeline=pipeline_show_pose,
        test_mode=True)

    custom_dataset = build_dataset(dataset_cfg)

    assert custom_dataset.test_mode is True
    print(f'length of dataset is: {len(custom_dataset)}')
    # exit()

    for image_id in tqdm(range(0, len(custom_dataset), 1)):
        data_i = custom_dataset[image_id]

        # image_i = data_i['img']
        # # convert to bgr and save with opencv
        # image_i = image_i.cpu().numpy()
        # for i in range(3):
        #     image_i[i] *= np.asarray([0.229, 0.224, 0.225])[i]
        #     image_i[i] += np.asarray([0.485, 0.456, 0.406])[i]
        # image_i = np.transpose(image_i, (1, 2, 0))
        # image_i_bgr = image_i[:, :, ::-1]
        # image_i_bgr = (image_i_bgr * 255).astype(np.uint8)

        image_path = data_i['img_metas'].data['image_file']
        # print(image_path)
        image_path_split = image_path.split('/')
        id_name = image_path_split[-3]
        image_name = image_path_split[-1]
        output_dir = os.path.join('/CT/EgoMocap/work/EgocentricFullBody/vis_results/hands_dataset', id_name)
        os.makedirs(output_dir, exist_ok=True)

        # cv2.imwrite('/CT/EgoMocap/work/EgocentricFullBody/vis_results/out_recovered.jpg', image_i_bgr)

        # joint 2d visualize
        image_i_bgr = cv2.imread(data_i['img_metas'].data['image_file'])
        joint_2d = data_i['img_metas'].data['joints_3d'][:, :2] * 4
        joint_2d[:, 0] += 128
        # image_i_bgr = draw_joints(joint_2d, image_i_bgr)
        image_i_bgr = draw_keypoints(joint_2d, image_i_bgr, radius=1)
        os.makedirs(os.path.join(output_dir, 'joint_2d'), exist_ok=True)
        joint_2d_output_path = os.path.join(output_dir, 'joint_2d', image_name)
        cv2.imwrite(joint_2d_output_path, image_i_bgr)

        # joint bbox visualize


        # hand 2d seg visualize
        left_hand_img_vis = data_i['img_metas'].data['left_hand_img']
        right_hand_img_vis = data_i['img_metas'].data['right_hand_img']
        left_hand_img_vis = left_hand_img_vis.cpu().numpy()
        right_hand_img_vis = right_hand_img_vis.cpu().numpy()
        left_hand_img_vis = np.transpose(left_hand_img_vis, (1, 2, 0)) * 255
        right_hand_img_vis = np.transpose(right_hand_img_vis, (1, 2, 0)) * 255
        left_hand_img_vis = left_hand_img_vis.astype(np.uint8)
        right_hand_img_vis = right_hand_img_vis.astype(np.uint8)

        os.makedirs(os.path.join(output_dir, 'left_hand_img'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'right_hand_img'), exist_ok=True)
        left_hand_img_output_path = os.path.join(output_dir, 'left_hand_img', image_name)
        right_hand_img_output_path = os.path.join(output_dir, 'right_hand_img', image_name)
        cv2.imwrite(left_hand_img_output_path, left_hand_img_vis)
        cv2.imwrite(right_hand_img_output_path, right_hand_img_vis)

        # put the hand image to hand pose estimation network

def run_hand_pose_model(left_hand_img, right_hand_img, img_metas):
    hand_pose_model = dict(
        type='EgoHandPose',
    )
    hand_pose_model = build_posenet(hand_pose_model)
    hand_pose_model.eval()

    output = hand_pose_model(left_hand_img, right_hand_img, img_metas=img_metas, return_loss=False)


if __name__ == '__main__':
    test_studio_with_hand_dataset()
    # test_mocap_studio_dataset()
    # test_mo2cap2_dataset()
