#  Copyright Jian Wang @ MPI-INF (c) 2023.

import pickle

import numpy as np
from tqdm import tqdm
from mmpose.utils.visualization.draw import draw_skeleton_with_chain

mo2cap2_chain = [[0, 1, 2, 3], [0, 4, 5, 6], [1, 7, 8, 9, 10], [4, 11, 12, 13, 14], [7, 11]]

def parse_egocentric_prediction(result_data):
    keypoints_pred_full_list = []
    keypoints_gt_full_list = []
    calibration_board_full_list = []
    ext_id_full_list = []
    seq_name_full_list = []
    image_file_full_list = []

    for i in tqdm(range(len(result_data))):
        result_data_i = result_data[i]
        image_file_list = []
        keypoints_gt_list = []
        calibration_board_list = []
        ext_id_list = []
        seq_name_list = []
        keypoints_pred = result_data_i['keypoints_pred']
        img_meta_list = result_data_i['img_metas']
        for img_meta_item in img_meta_list:
            image_file = img_meta_item['image_file']
            image_file_list.append(image_file)
            keypoints_gt = img_meta_item['keypoints_3d']
            keypoints_gt_list.append(keypoints_gt)

            calibration_board = img_meta_item['calib_board_pose']
            calibration_board_list.append(calibration_board)

            ext_id = img_meta_item['ext_id']
            ext_id_list.append(ext_id)

            seq_name = img_meta_item['seq_name']
            seq_name_list.append(seq_name)


        keypoints_pred_full_list.extend(keypoints_pred)
        keypoints_gt_full_list.extend(keypoints_gt_list)
        image_file_full_list.extend(image_file_list)
        calibration_board_full_list.extend(calibration_board_list)
        ext_id_full_list.extend(ext_id_list)
        seq_name_full_list.extend(seq_name_list)

    result_list = []
    for keypoint_pred, keypoint_gt, calibration_board, ext_id, seq_name, image_file in zip(
            keypoints_pred_full_list, keypoints_gt_full_list, calibration_board_full_list,
            ext_id_full_list, seq_name_full_list, image_file_full_list):
        result_i = dict(
            keypoints_pred=keypoint_pred,
            keypoints_gt=keypoint_gt,
            calibration_board=calibration_board,
            ext_id=ext_id,
            seq_name=seq_name,
            image_file=image_file
        )
        result_list.append(result_i)

    return result_list


def transform_new_body_pose(keypoints, calib_board_pose, hand_eye_calibration_matrix):
    fisheye_camera_transformation_matrix = np.linalg.inv(calib_board_pose).dot(
        np.linalg.inv(hand_eye_calibration_matrix))

    keypoints_homo = np.ones((len(keypoints), 4))
    keypoints_homo[:, :3] = keypoints
    global_keypoints_homo = fisheye_camera_transformation_matrix.dot(keypoints_homo.T).T
    transformed_pose = global_keypoints_homo[:, :3].astype(np.float32)

    return transformed_pose


def main(vis=False, save=False):
    hand_eye_calibration_file_path = r'hand_eye_calibration_0809.pkl'
    with open(hand_eye_calibration_file_path, 'rb') as f:
        hand_eye_calibration_matrix = pickle.load(f)
    estimated_egoentric_pose_file_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\vit_256x256_heatmap_3d_test\results.pkl'
    save_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\vit_256x256_heatmap_3d_test\results_global.pkl'

    with open(estimated_egoentric_pose_file_path, 'rb') as f:
        estimated_egopose_data = pickle.load(f)

    data_list = parse_egocentric_prediction(estimated_egopose_data)


    transformed_pose_pred_list = []
    transformed_pose_gt_list = []

    for data_i in tqdm(data_list):
        keypoints_pred = data_i['keypoints_pred']
        calib_board_pose = data_i['calibration_board']
        keypoints_gt = data_i['keypoints_gt']

        global_pose_pred = transform_new_body_pose(keypoints_pred, calib_board_pose,
                                                      hand_eye_calibration_matrix)

        global_pose_gt = transform_new_body_pose(keypoints_gt, calib_board_pose,
                                                        hand_eye_calibration_matrix)

        transformed_pose_pred_list.append(global_pose_pred)
        transformed_pose_gt_list.append(global_pose_gt)
        data_i['global_pose_pred'] = global_pose_pred
        data_i['global_pose_gt'] = global_pose_gt



    # split into sequences by seq_name
    seq_name_list = {}
    for data_i in tqdm(data_list):
        seq_name = data_i['seq_name']
        if seq_name not in seq_name_list.keys():
            seq_name_list[seq_name] = []
        seq_name_list[seq_name].append(data_i)
    # sort each sequence
    for seq_name in seq_name_list.keys():
        seq_name_list[seq_name] = sorted(seq_name_list[seq_name], key=lambda x: x['ext_id'])
    # save
    if save:
        with open(save_path, 'wb') as f:
            print('save data to: ', save_path)
            pickle.dump(seq_name_list, f)

    # vis
    if vis:
        import open3d
        coor = open3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh_list = [coor]

        for seq_name in seq_name_list.keys():
            data_seq = seq_name_list[seq_name]
            vis_data_list = data_seq[100: 500: 20]
            for data_i in vis_data_list:
                global_pose_pred = data_i['global_pose_pred']
                global_pose_gt = data_i['global_pose_gt']
                mesh_pred = draw_skeleton_with_chain(global_pose_pred, mo2cap2_chain)
                mesh_gt = draw_skeleton_with_chain(global_pose_gt, mo2cap2_chain,
                                                   joint_color=(1, 0, 0), bone_color=(1, 0, 0))
                mesh_list.append(mesh_pred)
                mesh_list.append(mesh_gt)
            open3d.visualization.draw_geometries(mesh_list)


if __name__ == '__main__':
    main(vis=False, save=True)
