#  Copyright Jian Wang @ MPI-INF (c) 2023.
import os
import pickle
from copy import deepcopy

import cv2

import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import torch
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated

fisheye_camera_path = '/CT/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
fisheye_model = FishEyeCameraCalibrated(fisheye_camera_path)

def keypoints_3d_to_pixel_aligned(keypoints_3d, joint_num=15):
    # joint_coord: [batch_size, joint_num, 3]
    # joint_coord_cam: [batch_size, joint_num, 3]
    # batch_size = joint_coord.shape[0]

    # calculate joint depth
    batch_size, joint_num, _ = keypoints_3d.shape
    joint_2d, joint_depth = fisheye_model.world2camera_pytorch_with_depth(keypoints_3d.view(batch_size * joint_num, 3))
    print(joint_2d.shape)
    # joint 2d resize to (64, 64)
    joint_2d = joint_2d.view(-1, joint_num, 2)
    joint_depth = joint_depth.view(-1, joint_num, 1)
    joint_2d[:, :, 0] -= 128
    joint_2d = joint_2d / 1024 * 64
    joint_depth = joint_depth / 2 * 64
    joint_coord_cam = torch.cat([joint_2d, joint_depth], dim=2)


    # joint_coord_xy = joint_coord[:, :, :2].view(batch_size * joint_num, 2)
    # joint_coord_z = joint_coord[:, :, 2].view(batch_size * joint_num)
    # joint_coord_cam = fisheye_model.camera2world_pytorch(joint_coord_xy, joint_coord_z)
    # joint_coord_cam = joint_coord_cam.view(batch_size, joint_num, 3)
    # joint_coord_cam = joint_coord_cam.contiguous()

    return joint_coord_cam

def visualize_3d_heatmap(heatmap, keypoints_3d_pixel_aligned, img, save_path=None):
    """Visualize 3D heatmap.

    Args:
        heatmap (np.ndarray[J, D, H, W]): 3D heatmap array.
        img (np.ndarray[H, W, C]): Image array.
        save_path (str): Path to save visualization results.
            Default: None.
    """

    num_joints, depth, height, width = heatmap.shape

    # heatmap = heatmap.reshape((num_joints, depth * height * width))
    # heatmap = softmax(heatmap, axis=1)
    # heatmap = heatmap.reshape((num_joints, depth, height, width))

    # convert the pred 3d body keypoints to pixel-aligned 3d representation
    for i in range(num_joints):
        heatmap_i = heatmap[i]
        heatmap_i = gaussian_filter(heatmap_i, sigma=3)
        heatmap[i] = heatmap_i

    # # get value at the keypoint
    heatmap_torch = torch.from_numpy(heatmap).view(num_joints, 1, depth, height, width)
    keypoints_3d_torch = keypoints_3d_pixel_aligned.reshape(num_joints, 1, 1, 1, 3)
    # no flip here!
    # keypoints_3d_torch = torch.flip(keypoints_3d_torch, dims=[4])
    # print(keypoints_3d_torch)
    # to (-1, 1)
    keypoints_3d_torch = keypoints_3d_torch / 64 * 2 - 1
    heatmap_value = torch.nn.functional.grid_sample(
        heatmap_torch, keypoints_3d_torch, align_corners=False)
    heatmap_value = heatmap_value.view(num_joints, 1)
    print(heatmap_value)

    keypoint_value_list = np.zeros((num_joints, 1))
    for i in range(num_joints):
        img_vis = deepcopy(img)
        heatmap_i = heatmap[i]
        keypoint_i = keypoints_3d_pixel_aligned[i].cpu().numpy()
        keypoint_i = np.round(keypoint_i).astype(np.int32)[::-1]
        # print(keypoint_i)

        keypoint_value = heatmap_i[keypoint_i[0], keypoint_i[1], keypoint_i[2]]
        keypoint_value_list[i, 0] = keypoint_value
        # print(f'joint_id {i}, keypoint_value: {keypoint_value}')
        # print(f'joint_id {i}, max_value: {np.max(heatmap_i)}')
        # heatmap_i = np.sum(heatmap_i, axis=0)
        # heatmap_i = heatmap_i / np.max(heatmap_i)
        # heatmap_i = np.uint8(heatmap_i * 255)
        # heatmap_i = cv2.resize(heatmap_i, (256, 256), interpolation=cv2.INTER_LINEAR)
        # img_vis = cv2.resize(img_vis, (256, 256), interpolation=cv2.INTER_LINEAR)
        # heatmap_i = cv2.applyColorMap(heatmap_i, cv2.COLORMAP_JET)
        # img_vis = cv2.addWeighted(img_vis, 0.5, heatmap_i, 0.5, 0)
        # cv2.imwrite(save_path + f'_{i}.jpg', img_vis)
    # norm
    # keypoint_value_list = keypoint_value_list / np.max(keypoint_value_list)
    print(keypoint_value_list)



def visualize_results_heatmap(result_path):
    with open(result_path, 'rb') as f:
        result_data = pickle.load(f)

    image_file_full_list = []
    keypoints_pred_full_list = []
    heatmap_full_list = []

    for i in tqdm(range(len(result_data))):
        result_data_i = result_data[i]
        image_file_list = []
        keypoints_pred = result_data_i['keypoints_pred']
        heatmap_pred = result_data_i['heatmap']
        img_meta_list = result_data_i['img_metas']
        for img_meta_item in img_meta_list:
            image_file = img_meta_item['image_file']
            image_file_list.append(image_file)

        heatmap_full_list.extend(heatmap_pred)
        image_file_full_list.extend(image_file_list)
        keypoints_pred_full_list.extend(keypoints_pred)

    keypoints_pred_full_list = torch.as_tensor(keypoints_pred_full_list).float()
    print(keypoints_pred_full_list.shape)
    keypoints_pixel_aligned = keypoints_3d_to_pixel_aligned(keypoints_pred_full_list)

    for heatmap_3d, keypoints_pred, image_file in zip(heatmap_full_list, keypoints_pixel_aligned,
                                                      image_file_full_list):
        image = cv2.imread(image_file)
        image = image[:, 128: -128, :]
        save_path = '/CT/EgoMocap/work/EgocentricFullBody/vis_results/heatmap_3d'
        print(image_file)
        os.makedirs(save_path, exist_ok=True)
        visualize_3d_heatmap(heatmap_3d, keypoints_pred,
                             image, save_path=os.path.join(save_path, os.path.basename(image_file)))

if __name__ == '__main__':
    if os.name == 'nt':
        result_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\vit_256x256_heatmap_3d_sample\results.pkl'
    else:
        result_path = r'/CT/EgoMocap/work/EgocentricFullBody/work_dirs/vit_256x256_heatmap_3d_sample/results.pkl'
    visualize_results_heatmap(result_path)