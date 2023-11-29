import pickle
from tqdm import tqdm
import os
import numpy as np
import cv2
from copy import deepcopy
from mmpose.utils.visualization.draw import draw_keypoints, draw_bbox
from mmpose.data.keypoints_mapping.renderpeople import render_people_joint_names


def bbox_from_joints(joint_list, image_w=1280, image_h=1024):
    min_xy = np.min(joint_list, axis=0)
    min_xy[min_xy < 0] = 0
    max_xy = np.max(joint_list, axis=0)
    if max_xy[0] > image_w:
        max_xy[0] = image_w
    if max_xy[1] > image_h:
        max_xy[1] = image_h

    center_xy = (min_xy + max_xy) / 2

    radius = np.linalg.norm(max_xy - min_xy) / 2

    return {'min_xy': min_xy, 'max_xy': max_xy, 'center': center_xy, 'radius': radius}


def bbox_to_square(bbox, min_bbox_side=32, image_w=1280, image_h=1024):
    min_xy = bbox['min_xy']
    max_xy = bbox['max_xy']

    max_side = np.max(max_xy - min_xy)
    max_side = max(max_side, min_bbox_side)
    max_side = np.asarray([max_side, max_side])
    center_xy = bbox['center']

    new_min_xy = center_xy - max_side
    new_max_xy = center_xy + max_side

    # this will break the square feature
    new_min_xy[new_min_xy < 0] = 0
    if new_max_xy[0] > image_w:
        new_max_xy[0] = image_w
    if new_max_xy[1] > image_h:
        new_max_xy[1] = image_h

    radius = np.linalg.norm(new_max_xy - new_min_xy) / 2

    return {'min_xy': new_min_xy, 'max_xy': new_max_xy, 'center': center_xy, 'radius': radius}

if __name__ == '__main__':

    # result_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\hrnet_256x256_3d_train_head_1\results.pkl'
    result_path = r'/CT/EgoMocap/work/EgocentricFullBody/work_dirs/hrnet_256x256_full_body_2d_test/results.pkl'

    out_dir = '/CT/EgoMocap/work/EgocentricFullBody/vis_results/hrnet_256x256_full_body_2d_test'
    os.makedirs(out_dir, exist_ok=True)

    with open(result_path, 'rb') as f:
        result_data = pickle.load(f)

    image_path_list = result_data['image_file']
    joint_2d_list = result_data['joints_2d_pred']

    for i in tqdm(range(0, len(image_path_list), 100)):
        image_path = image_path_list[i]
        image_name = os.path.split(image_path)[1]
        # print(image_name)
        img = cv2.imread(image_path)
        original_img = deepcopy(img)

        joint_i = joint_2d_list[i] * 1024 / 64
        joint_i[:, 0] += 128

        left_hand_joints = joint_i[25: 40]
        left_hand_joints = np.concatenate([left_hand_joints, joint_i[20:21]], axis=0)
        right_hand_joints = joint_i[40: 55]
        right_hand_joints = np.concatenate([right_hand_joints, joint_i[21:22]], axis=0)

        # make bounding box from joints
        left_hand_bbox = bbox_from_joints(left_hand_joints)

        right_hand_bbox = bbox_from_joints(right_hand_joints)

        # fix the bbox (to square and make it larger)
        left_hand_bbox = bbox_to_square(left_hand_bbox, min_bbox_side=32)
        right_hand_bbox = bbox_to_square(right_hand_bbox, min_bbox_side=32)

        img = draw_bbox(img, left_hand_bbox['min_xy'], left_hand_bbox['max_xy'])
        img = draw_bbox(img, right_hand_bbox['min_xy'], right_hand_bbox['max_xy'])
        img = draw_keypoints(joint_i, img)

        out_path = os.path.join(out_dir, image_name)

        cv2.imwrite(out_path, img)

        left_hand_bbox_save_path = os.path.join(out_dir, f'left_{image_name}')
        right_hand_bbox_save_path = os.path.join(out_dir, f'right_{image_name}')

        left_hand_img = original_img[int(left_hand_bbox['min_xy'][1]): int(left_hand_bbox['max_xy'][1]),
                        int(left_hand_bbox['min_xy'][0]): int(left_hand_bbox['max_xy'][0]), :]

        right_hand_img = original_img[int(right_hand_bbox['min_xy'][1]): int(right_hand_bbox['max_xy'][1]),
                         int(right_hand_bbox['min_xy'][0]): int(right_hand_bbox['max_xy'][0]), :]

        cv2.imwrite(left_hand_bbox_save_path, left_hand_img)
        cv2.imwrite(right_hand_bbox_save_path, right_hand_img)

