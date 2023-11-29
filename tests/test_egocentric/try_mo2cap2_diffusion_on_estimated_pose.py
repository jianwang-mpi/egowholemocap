import os

import numpy as np
import open3d
import torch

from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines import Collect
from mmpose.models.diffusion_mdm.data_loaders.humanml.scripts.motion_process import (
    recover_from_ric,
)

if os.name == "nt":
    seq_path = r"Z:\EgoMocap\work\EgocentricFullBody\work_dirs\vit_256x256_heatmap_3d_test\results_global.pkl"
    std_path = r"Z:/EgoMocap/work/motion-diffusion-model\dataset\HumanML3D\Std.npy"
    mean_path = r"Z:\EgoMocap\work\motion-diffusion-model\dataset\HumanML3D\Mean.npy"
    smplx_model_dir = r"Z:\EgoMocap\work\EgocentricFullBody\human_models\smplx_new"
    mdm_model_path = r"Z:\EgoMocap\work\motion-diffusion-model\save\humanml_trans_enc_512\model000200000.pt"
    diffusion_result_save_dir = (
        r"Z:\EgoMocap\work\EgocentricFullBody\vis_results\diffusion_results"
    )
else:
    seq_path = r"/CT/EgoMocap/work/EgocentricFullBody/work_dirs/vit_256x256_heatmap_3d_test/results_global.pkl"
    std_path = r"/CT/EgoMocap/work/motion-diffusion-model/dataset/HumanML3D/Std.npy"
    mean_path = r"/CT/EgoMocap/work/motion-diffusion-model/dataset/HumanML3D/Mean.npy"
    smplx_model_dir = r"/CT/EgoMocap/work/EgocentricFullBody/human_models/smplx_new"
    mdm_model_path = r"/CT/EgoMocap/work/motion-diffusion-model/save/humanml_trans_enc_512/model000200000.pt"
    diffusion_result_save_dir = (
        r"/CT/EgoMocap/work/EgocentricFullBody/vis_results/diffusion_results"
    )


def get_mo2cap2_dataset():

    pipeline = [
        Collect(
            keys=["data", "mask", "lengths", "mean", "std", "gt", "image_names"],
            meta_keys=[],
        )
    ]

    path = seq_path

    dataset_cfg = dict(
        type="Mo2Cap2MotionDataset",
        path_dict=path,
        frame_rate=20,
        seq_len=196,
        std_path=std_path,
        mean_path=mean_path,
        pipeline=pipeline,
        test_mode=True,
    )

    amass_dataset = build_dataset(dataset_cfg)
    print(f"length of dataset is: {len(amass_dataset)}")
    return amass_dataset


def get_data(mo2cap2_dataset, data_id, vis=False):
    data_i = mo2cap2_dataset[data_id]

    seq_i = data_i["data"]
    seq_gt_i = data_i["gt"]
    # convert to bgr and save with opencv
    print(seq_i.shape)
    length_i = data_i["lengths"]
    mask_i = data_i["mask"]
    print(length_i)
    print(mask_i.shape)

    # visualize the dataset with open3d

    from mmpose.datasets.datasets.diffusion.keypoints_to_hml3d import recover_from_ric
    from mmpose.utils.visualization.draw import draw_skeleton_with_chain
    from mmpose.models.diffusion_mdm.data_loaders.humanml.utils import paramUtil

    joint_chain = paramUtil.t2m_kinematic_chain
    if isinstance(seq_i, np.ndarray):
        seq_i = torch.asarray(seq_i).float()
    seq_i = seq_i * mo2cap2_dataset.std + mo2cap2_dataset.mean
    joint_location_seq = recover_from_ric(seq_i, 22)
    if not isinstance(joint_location_seq, np.ndarray):
        joint_location_seq = joint_location_seq.numpy()

    data_i["motions"] = joint_location_seq

    if vis:
        coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
        for pose_i in joint_location_seq:
            skeleton_i = draw_skeleton_with_chain(pose_i, joint_chain)
            open3d.visualization.draw_geometries([skeleton_i, coord])

    return data_i


mo2cap2_dataset = get_mo2cap2_dataset()
from mmpose.models.diffusion_mdm.edit_mo2cap2 import Mo2Cap2Diffusion

mdm_edit = Mo2Cap2Diffusion(model_path=mdm_model_path, max_frames=196).cuda()
mdm_edit.eval()


def main(data_id):

    data_i = get_data(mo2cap2_dataset, data_id, vis=False)
    for k, v in data_i.items():
        if k in ["data", "mask", "lengths", "mean", "std", "motions", "gt"]:
            data_i[k] = torch.asarray(v).unsqueeze(0).cuda().float()

    mean = data_i["mean"]
    std = data_i["std"]

    with torch.no_grad():
        all_motions = mdm_edit(**data_i)

    out_motions = all_motions["motions"]
    input_motions = all_motions["input_motions"]
    input_motions = input_motions.permute(0, 2, 3, 1)
    input_motions = input_motions * std + mean  # this should be in the dataset!
    input_motions = recover_from_ric(input_motions, 22)
    # sample shape: (batch_size, 1, max_frames, 22, 3)
    input_motions = input_motions.view(-1, *input_motions.shape[2:])
    input_motions = input_motions.cpu().numpy()

    gt = data_i["gt"].cpu().numpy()
    image_names = data_i["image_names"]
    print(f"image starts from\n{image_names[0]}\n to\n{image_names[-1]}\n")

    save_dir = os.path.join(diffusion_result_save_dir, str(data_id))
    print(f"save at: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    mdm_edit.save_result(out_motions, input_motions, image_names, save_dir)

    mdm_edit.show_result(results=out_motions, inputs=input_motions, save_dir=save_dir)

    # calculate mpjpe
    from mmpose.core.evaluation.pose3d_eval import keypoint_mpjpe

    out_motions = np.squeeze(out_motions)
    out_mo2cap2_pose = np.zeros((len(out_motions), 15, 3))
    out_mo2cap2_pose[:, mo2cap2_dataset.dst_idxs] = out_motions[
        :, mo2cap2_dataset.model_idxs
    ]
    input_motions = np.squeeze(input_motions)
    input_mo2cap2_pose = np.zeros((len(input_motions), 15, 3))
    input_mo2cap2_pose[:, mo2cap2_dataset.dst_idxs] = input_motions[
        :, mo2cap2_dataset.model_idxs
    ]
    gt = np.squeeze(gt)
    mask = np.ones((gt.shape[0], gt.shape[1])).astype(bool)

    pred_mpjpe = keypoint_mpjpe(out_mo2cap2_pose, gt, mask=mask, alignment="procrustes")
    input_mpjpe = keypoint_mpjpe(
        input_mo2cap2_pose, gt, mask=mask, alignment="procrustes"
    )

    print("pred mpjpe: ", pred_mpjpe)
    print("input mpjpe: ", input_mpjpe)
    return pred_mpjpe, input_mpjpe


if __name__ == "__main__":
    pa_mpjpe_list = []
    input_mpjpe_list = []
    for i in range(69):
        if i != 57:
            continue
        pred_mpjpe, input_mpjpe = main(i)
        pa_mpjpe_list.append(pred_mpjpe)
        input_mpjpe_list.append(input_mpjpe)
    print("pa mpjpe: ", np.mean(pa_mpjpe_list))
    print("input mpjpe: ", np.mean(input_mpjpe_list))
