#  Copyright Jian Wang @ MPI-INF (c) 2023.

from npybvh.bvh import Bvh
import numpy as np
from tqdm import tqdm
import pickle
from mmpose.data.keypoints_mapping.studio import studio_original_joint_names

anim = Bvh()

def parse_bvh_file(bvh_file_path, start_frame=0, input_frame_rate=25, output_frame_rate=25):
    anim.parse_file(bvh_file_path)
    gt_pose_seq = []
    print(anim.frames)
    print(anim.joint_names())
    joint_name_list = list(anim.joint_names())
    egocentric_joints = [joint_name_list.index(jt_name) for jt_name in studio_original_joint_names]
    step = round(input_frame_rate / output_frame_rate)
    for frame in tqdm(range(start_frame, anim.frames, step)):
        positions, rotations = anim.frame_pose(frame)

        positions = positions[egocentric_joints]
        positions = positions / 1000
        gt_pose_seq.append(positions)

    gt_pose_seq = np.asarray(gt_pose_seq)
    return gt_pose_seq

def parse_bvh_file_and_save(bvh_file_path, output_file_path, start_frame=0, input_frame_rate=25, output_frame_rate=25):
    gt_pose_seq = parse_bvh_file(bvh_file_path, start_frame, input_frame_rate, output_frame_rate)
    if output_file_path is not None:
        with open(output_file_path, 'wb') as f:
            pickle.dump(gt_pose_seq, f)


if __name__ == '__main__':
    id_name = 'diogo1'
    parse_bvh_file_and_save(fr'X:\ScanNet\work\05-08-22\{id_name}\unknown.bvh',
               fr'data/05-08-2022/{id_name}/pose_gt_with_hand_new.pkl',
               start_frame=0, input_frame_rate=25, output_frame_rate=25)
