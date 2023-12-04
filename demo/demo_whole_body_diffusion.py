import os
import pickle

import torch
from mmcv.runner import load_checkpoint
from natsort import natsorted

from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import (Collect,
                                       PreProcessHandMotion, SplitGlobalSMPLXJoints,
                                       PreProcessMo2Cap2BodyMotion, ExtractConfidence)
from mmpose.models.diffusion_hands.refine_mo2cap2_hands_with_uncertainty import RefineEdgeDiffusionHandsUncertainty




def get_motion_dataset():
    seq_len = 196
    normalize = True
    pipeline = [
        SplitGlobalSMPLXJoints(smplx_joint_name='ego_smplx_joints'),
        PreProcessHandMotion(normalize=normalize,
                             mean_std_path=mean_std_path),
        PreProcessMo2Cap2BodyMotion(normalize=normalize,
                                    mean_std_path=mean_std_path),
        ExtractConfidence(confidence_name='human_body_confidence', target_range=(0.995, 1)),
        ExtractConfidence(confidence_name='left_hand_confidence', target_range=(0.995, 1)),
        ExtractConfidence(confidence_name='right_hand_confidence', target_range=(0.995, 1)),
        Collect(keys=['mo2cap2_body_features', 'left_hand_features', 'right_hand_features',
                      'human_body_confidence', 'left_hand_confidence', 'right_hand_confidence',
                      'processed_left_hand_keypoints_3d', 'processed_right_hand_keypoints_3d'],
                meta_keys=['image_path'])
    ]

    dataset_cfg = dict(
        type='FullBodyEgoMotionEvalDataset',
        data_pkl_path=network_pred_seq_path,
        seq_len=seq_len,
        skip_frames=seq_len,
        pipeline=pipeline,
        split_sequence=True,
        test_mode=True,
    )

    fullbody_motion_test_dataset = build_dataset(dataset_cfg)
    print(f"length of dataset is: {len(fullbody_motion_test_dataset)}")
    return fullbody_motion_test_dataset


def run_diffusion():
    full_body_motion_dataset = get_motion_dataset()
    data_length = len(full_body_motion_dataset)
    result = []

    full_body_pose_diffusion_refiner = RefineEdgeDiffusionHandsUncertainty(seq_len=196).cuda()
    full_body_pose_diffusion_refiner.eval()
    load_checkpoint(full_body_pose_diffusion_refiner, diffusion_model_path, map_location='cpu', strict=True)
    for data_id in range(data_length):
        data_i = full_body_motion_dataset[data_id]

        data_i = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v for k, v in data_i.items()}

        with torch.no_grad():
            diffusion_results = full_body_pose_diffusion_refiner(**data_i)

        pred_motion_result = get_pred_motion(diffusion_results)
        # split the data into lists
        seq_len = pred_motion_result['left_hand_pred_motion'].shape[0]
        pred_motion_result_list = []
        for i in range(seq_len):
            pred_motion_result_i = {}
            for key in pred_motion_result.keys():
                pred_motion_result_i[key] = pred_motion_result[key][i]
            pred_motion_result_list.append(pred_motion_result_i)
        result.extend(pred_motion_result_list)

    result = natsorted(result, key=lambda x: x['image_path'])
    return result


def get_pred_motion(diffusion_results):
    normalize = True

    full_body_motion_sequence_list = diffusion_results['sample'].cpu().numpy()
    mo2cap2_motion_sequence_list = full_body_motion_sequence_list[:, :, :15 * 3]
    left_hand_motion_sequence_list = full_body_motion_sequence_list[:, :, 15 * 3: 15 * 3 + 21 * 3]
    right_hand_motion_sequence_list = full_body_motion_sequence_list[:, :, 15 * 3 + 21 * 3:]

    full_body_input_motion_list = diffusion_results['full_body_features'].cpu().numpy()
    mo2cap2_input_motion_list = full_body_input_motion_list[:, :, :15 * 3]
    left_hand_input_motion_list = full_body_input_motion_list[:, :, 15 * 3: 15 * 3 + 21 * 3]
    right_hand_input_motion_list = full_body_input_motion_list[:, :, 15 * 3 + 21 * 3:]

    image_path_list = diffusion_results['img_metas'].data['image_path']

    with open(mean_std_path, 'rb') as f:
        global_aligned_mean_std = pickle.load(f)

    if normalize:
        left_hand_mean = global_aligned_mean_std['left_hand_mean']
        left_hand_std = global_aligned_mean_std['left_hand_std']
        left_hand_motion_sequence_list = left_hand_motion_sequence_list * left_hand_std + left_hand_mean
        left_hand_input_motion_list = left_hand_input_motion_list * left_hand_std + left_hand_mean
        right_hand_mean = global_aligned_mean_std['right_hand_mean']
        right_hand_std = global_aligned_mean_std['right_hand_std']
        right_hand_motion_sequence_list = right_hand_motion_sequence_list * right_hand_std + right_hand_mean
        right_hand_input_motion_list = right_hand_input_motion_list * right_hand_std + right_hand_mean
        mo2cap2_body_mean = global_aligned_mean_std['mo2cap2_body_mean']
        mo2cap2_body_std = global_aligned_mean_std['mo2cap2_body_std']
        mo2cap2_motion_sequence_list = mo2cap2_motion_sequence_list * mo2cap2_body_std + mo2cap2_body_mean
        mo2cap2_input_motion_list = mo2cap2_input_motion_list * mo2cap2_body_std + mo2cap2_body_mean

    left_hand_pred_motion = left_hand_motion_sequence_list.reshape(-1, 21, 3)
    right_hand_pred_motion = right_hand_motion_sequence_list.reshape(-1, 21, 3)
    mo2cap2_pred_motion = mo2cap2_motion_sequence_list.reshape(-1, 15, 3)
    left_hand_pred_motion[:, 0] *= 0
    right_hand_pred_motion[:, 0] *= 0

    left_hand_input_motion = left_hand_input_motion_list.reshape(-1, 21, 3)
    left_hand_input_motion[:, 0] *= 0
    right_hand_input_motion = right_hand_input_motion_list.reshape(-1, 21, 3)
    right_hand_input_motion[:, 0] *= 0
    mo2cap2_input_motion = mo2cap2_input_motion_list.reshape(-1, 15, 3)

    assert len(image_path_list) == len(left_hand_pred_motion)

    result = {'left_hand_pred_motion': left_hand_pred_motion,
              'left_hand_input_motion': left_hand_input_motion,
              'right_hand_pred_motion': right_hand_pred_motion,
              'right_hand_input_motion': right_hand_input_motion,
              'mo2cap2_pred_motion': mo2cap2_pred_motion,
              'mo2cap2_input_motion': mo2cap2_input_motion,
              'image_path': image_path_list
              }
    return result


if __name__ == '__main__':
    # network_pred_seq_path = r'work_dirs/egowholebody_single_demo/outputs.pkl'
    mean_std_path = r'dataset_files/ego_mean_std.pkl'
    diffusion_model_path = r'checkpoints/diffusion_denoiser.pth'
    diffusion_result_save_dir = r"work_dirs/egowholebody_diffusion_demo"

    import argparse

    parser = argparse.ArgumentParser(description='visualize single frame whole body result')
    parser.add_argument('--pred_path', type=str, required=True, help='prediction output pkl file path')
    args = parser.parse_args()
    network_pred_seq_path = args.pred_path
    result_list = run_diffusion()
    os.makedirs(diffusion_result_save_dir, exist_ok=True)
    with open(os.path.join(diffusion_result_save_dir, 'outputs.pkl'), 'wb') as f:
        pickle.dump(result_list, f)
