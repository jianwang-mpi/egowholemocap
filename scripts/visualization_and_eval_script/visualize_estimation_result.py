import pickle

import cv2
from tqdm import tqdm
import open3d.visualization
import numpy as np
from mmpose.core.evaluation.pose3d_eval import keypoint_mpjpe
from mmpose.utils.visualization.draw import draw_joints

from mmpose.utils.visualization.skeleton import Skeleton



# result_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\vit_256x256_heatmap_3d_finetune\results.pkl'
result_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\hrnet_256x256_3d_train_head_1\results.pkl'
# result_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\hrnet_256x256_3d_eval_hiro\results_hiro.pkl'
fisheye_camera_path = r'Z:\EgoMocap\work\EgocentricFullBody\mmpose\utils\fisheye_camera\fisheye.calibration_01_12.json'
skeleton = Skeleton(fisheye_camera_path)

with open(result_path, 'rb') as f:
    result_data = pickle.load(f)

keypoints_pred_full_list = []
keypoints_gt_full_list = []
image_file_full_list = []

for i in tqdm(range(len(result_data))):
    result_data_i = result_data[i]
    image_file_list = []

    keypoints_gt_list = []
    keypoints_pred = result_data_i['keypoints_pred']
    img_meta_list = result_data_i['img_metas']
    for img_meta_item in img_meta_list:
        image_file = img_meta_item['image_file']
        image_file_list.append(image_file)
        keypoints_gt = img_meta_item['keypoints_3d']
        keypoints_gt_list.append(keypoints_gt)

    keypoints_pred_full_list.extend(keypoints_pred)
    keypoints_gt_full_list.extend(keypoints_gt_list)
    image_file_full_list.extend(image_file_list)

visualize=False

if visualize:
    for i in range(0, len(image_file_full_list), 100):
        if i < 2000:
            continue
        print(i)

        image_name = image_file_full_list[i]
        keypoints_pred = keypoints_pred_full_list[i]
        keypoints_gt = keypoints_gt_full_list[i]
        print(image_name)
        image_name = image_name.replace('/CT', 'Z:').replace('/HPS', 'X:')
        img = cv2.imread(image_name)
        # img = img[10: -80, 490:-440, :]

        # draw projected image
        keypoints_pred_2d = skeleton.camera.world2camera(keypoints_pred)
        img = draw_joints(keypoints_pred_2d, img)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        joint_mesh = skeleton.joints_2_mesh(keypoints_pred)
        joints_gt_mesh = skeleton.joints_2_mesh(keypoints_gt, joint_color=(1, 0, 0), bone_color=(1, 0, 0))
        open3d.visualization.draw_geometries([joint_mesh, joints_gt_mesh])
        # open3d.visualization.draw_geometries([joint_mesh])

cal_error = True

if cal_error:
    keypoints_pred_full_list = np.asarray(keypoints_pred_full_list)
    keypoints_gt_full_list = np.asarray(keypoints_gt_full_list)
    N, K, C = keypoints_pred_full_list.shape
    gt_joints_visible = np.ones((N, K)).astype(bool)
    res = keypoint_mpjpe(keypoints_pred_full_list, keypoints_gt_full_list, gt_joints_visible, alignment='none')
    print(res)
