#  Copyright Jian Wang @ MPI-INF (c) 2023.

import os
import numpy as np
import pickle
from parse_bvh import parse_bvh_file


def transform_new_body_pose(ext_id, calib_board_pose, hand_eye_calibration_matrix, ext_pose_gt):
    fisheye_camera_transformation_matrix = np.linalg.inv(calib_board_pose).dot(
        np.linalg.inv(hand_eye_calibration_matrix))

    pose_gt_homo = np.ones((len(ext_pose_gt), 4))
    pose_gt_homo[:, :3] = ext_pose_gt
    pose_gt_homo = np.linalg.inv(fisheye_camera_transformation_matrix).dot(pose_gt_homo.T).T
    transformed_pose_gt = pose_gt_homo[:, :3].astype(np.float32)

    return {'ext_id': ext_id,
            'calib_board_pose': calib_board_pose,
            'ego_camera_pose': fisheye_camera_transformation_matrix,
            'ego_pose_gt': transformed_pose_gt,
            'ext_pose_gt': ext_pose_gt}

def main():
    seq_name = 'diogo2'
    hand_eye_calibration_file_path = r'hand_eye_calibration_0809.pkl'
    with open(hand_eye_calibration_file_path, 'rb') as f:
        hand_eye_calibration_matrix = pickle.load(f)
    bvh_file_path = fr'X:\ScanNet\work\25-08-22\{seq_name}\unknown.bvh'
    previous_local_gt_path = fr'X:\ScanNet\work\egocentric_view\25082022\{seq_name}\local_pose_gt_with_hand_old.pkl'

    save_path = fr'X:\ScanNet\work\egocentric_view\25082022\{seq_name}\local_pose_gt_with_hand.pkl'

    with open(previous_local_gt_path, 'rb') as f:
        previous_local_gt = pickle.load(f)
    ext_id_list = previous_local_gt['ext_id']
    calib_board_list = previous_local_gt['calib_board_pose']

    ext_pose_gt_list = parse_bvh_file(bvh_file_path, start_frame=0, input_frame_rate=25, output_frame_rate=25)

    assert len(calib_board_list) == len(ext_pose_gt_list)

    result_list = []

    for ext_id in range(len(ext_id_list)):
        assert ext_id == ext_id_list[ext_id]

        calib_board_pose = calib_board_list[ext_id]
        ext_pose_gt = ext_pose_gt_list[ext_id]
        transformed_pose_gt = transform_new_body_pose(ext_id, calib_board_pose,
                                                      hand_eye_calibration_matrix, ext_pose_gt)

        result_list.append(transformed_pose_gt)

    # save
    with open(save_path, 'wb') as f:
        pickle.dump(result_list, f)


if __name__ == '__main__':
    main()