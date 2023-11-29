#  Copyright Jian Wang @ MPI-INF (c) 2023.

import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm

from mmpose.utils.visualization.draw import draw_keypoints


def create_bbox_from_joints(joints_2d):
    x_min = joints_2d[:, 0].min()
    x_max = joints_2d[:, 0].max()
    y_min = joints_2d[:, 1].min()
    y_max = joints_2d[:, 1].max()

    return x_min, y_min, x_max, y_max

def create_bbox_from_joints_with_scale(joints_2d, scale=1.3):
    x_min, y_min, x_max, y_max = create_bbox_from_joints(joints_2d)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    max_side = max(width, height)
    width = max_side * scale
    height = max_side * scale

    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2

    return x_min, y_min, x_max, y_max


def calculate_iou(bbox1, bbox2):
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2

    x_min = max(x_min1, x_min2)
    y_min = max(y_min1, y_min2)
    x_max = min(x_max1, x_max2)
    y_max = min(y_max1, y_max2)

    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    union = (x_max1 - x_min1) * (y_max1 - y_min1) + (x_max2 - x_min2) * (y_max2 - y_min2) - intersection

    return intersection / union


def eval_hand_detection(drawbbox=False):
    # result_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\hrnet_256x256_3d_train_head_1\results.pkl'
    result_path = r'/CT/EgoMocap/work/EgocentricFullBody/work_dirs/hrnet_full_body_test_finetune_only_hand/results.pkl'

    out_dir = '/CT/EgoMocap/work/EgocentricFullBody/vis_results/hand_detection'
    os.makedirs(out_dir, exist_ok=True)

    with open(result_path, 'rb') as f:
        result_data = pickle.load(f)

    image_path_list = result_data['image_file']
    joint_2d_pred = result_data['joints_2d_pred']
    joint_2d_gt = result_data['joints_2d_gt']

    iou_list = []

    for i in tqdm(range(len(image_path_list))):
        image_path = image_path_list[i]
        image_name = os.path.split(image_path)[1]

        joint_i = joint_2d_pred[i] * 1024 / 64
        joint_i[:, 0] += 128
        joint_gt_i = joint_2d_gt[i] * 1024 / 256
        joint_gt_i[:, 0] += 128

        joint_i_left_hand = joint_i[0:21]
        joint_i_right_hand = joint_i[21:42]

        joint_i_gt_left_hand = joint_gt_i[0:21]
        joint_i_gt_right_hand = joint_gt_i[21:42]
        # print(joint_i_gt_right_hand)

        bbox_pred_left_hand = create_bbox_from_joints_with_scale(joint_i_left_hand, scale=1.3)
        bbox_pred_right_hand = create_bbox_from_joints_with_scale(joint_i_right_hand, scale=1.3)
        bbox_gt_left_hand = create_bbox_from_joints_with_scale(joint_i_gt_left_hand, scale=1.2)
        bbox_gt_right_hand = create_bbox_from_joints_with_scale(joint_i_gt_right_hand, scale=1.2)

        if i % 100 == 0:
            if drawbbox is False:
                # print(joint_i)
                # print(joint_gt_i)
                img = cv2.imread(image_path)
                img = draw_keypoints(joint_i, img)
                img = draw_keypoints(joint_gt_i, img, color=(0, 255, 0))

                out_path = os.path.join(out_dir, image_name)
                cv2.imwrite(out_path, img)
            else:
                img = cv2.imread(image_path)
                img = cv2.rectangle(img, (int(bbox_pred_left_hand[0]), int(bbox_pred_left_hand[1])),
                                    (int(bbox_pred_left_hand[2]), int(bbox_pred_left_hand[3])),
                                    (0, 0, 255), 2)
                img = cv2.rectangle(img, (int(bbox_pred_right_hand[0]), int(bbox_pred_right_hand[1])),
                                    (int(bbox_pred_right_hand[2]), int(bbox_pred_right_hand[3])),
                                    (0, 0, 255), 2)
                img = cv2.rectangle(img, (int(bbox_gt_left_hand[0]), int(bbox_gt_left_hand[1])),
                                    (int(bbox_gt_left_hand[2]), int(bbox_gt_left_hand[3])),
                                    (0, 255, 0), 2)
                img = cv2.rectangle(img, (int(bbox_gt_right_hand[0]), int(bbox_gt_right_hand[1])),
                                    (int(bbox_gt_right_hand[2]), int(bbox_gt_right_hand[3])),
                                    (0, 255, 0), 2)

                out_path = os.path.join(out_dir, image_name)
                cv2.imwrite(out_path, img)

        iou = calculate_iou(bbox_pred_left_hand, bbox_gt_left_hand) + \
              calculate_iou(bbox_pred_right_hand, bbox_gt_right_hand)
        iou = iou / 2
        iou_list.append(iou)
    average_iou = np.mean(iou_list)
    print('average_iou: ', average_iou)


if __name__ == '__main__':
    eval_hand_detection(drawbbox=True)
