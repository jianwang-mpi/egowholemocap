import os.path
from copy import deepcopy

import cv2
import numpy as np
import torch
from torch.nn import DataParallel
from tqdm import tqdm

from mmpose.datasets.builder import build_dataset
from mmpose.datasets.pipelines.ego_hand_crop_transform import CropHandImageFisheye
from mmpose.models.builder import build_posenet
from mmpose.datasets.pipelines import (LoadImageFromFile, CropCircle, Generate2DPose,
                                       CropImage, ResizeImage, Generate2DPoseConfidence, ToTensor, NormalizeTensor,
                                       TopDownGenerateTarget, Collect, Generate2DHandPose, CropHandImage,
                                       ResizeImageWithName, RGB2BGRHand, ToTensorHand)
from mmpose.utils.visualization.draw import draw_keypoints
from mmpose.models.ego_hand_pose_estimation.utils.human_models import mano

def test_studio_with_hand_dataset(image_id):
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
        num_joints=73,
        camera_param_path=fisheye_camera_path,
        joint_type='renderpeople',
        image_size=[img_res, img_res],
        hand_image_size=[img_res, img_res],
        heatmap_size=(64, 64),
        joint_weights=None,
        use_different_joint_weights=False,
    )

    pipeline_show_pose = [
        LoadImageFromFile(),
        CropCircle(img_h=1024, img_w=1280),
        Generate2DPose(fisheye_model_path=fisheye_camera_path),
        Generate2DHandPose(fisheye_model_path=fisheye_camera_path),
        CropHandImageFisheye(fisheye_camera_path, input_img_h=1024, input_img_w=1280,
                             crop_img_size=256, enlarge_scale=1.4),
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
        Collect(keys=['img', 'target', 'target_weight', 'keypoints_2d_visible'],
                meta_keys=['image_file', 'keypoints_3d', 'joints_3d', 'joints_3d_visible',
                           'left_hand_keypoints_3d', 'right_hand_keypoints_3d',
                           'left_hand_img', 'right_hand_img']),
    ]

    dataset_cfg = dict(
        type=dataset_name,
        path_dict=path_dict,
        data_cfg=data_cfg,
        pipeline=pipeline_show_pose,
        test_mode=True)

    custom_dataset = build_dataset(dataset_cfg)

    assert custom_dataset.test_mode is True
    print(f'length of dataset is: {len(custom_dataset)}')

    data_i = custom_dataset[image_id]

    # visualize 2d heatmaps
    heatmap_2d = data_i['target']
    heatmap_2d_visible = data_i['keypoints_2d_visible']
    print(heatmap_2d.shape)

    heatmap_save_dir = os.path.join(output_dir, f'heatmap_{image_id}')
    os.makedirs(heatmap_save_dir, exist_ok=True)



    image_path = data_i['img_metas'].data['image_file']
    # print(image_path)
    image_path_split = image_path.split('/')
    id_name = image_path_split[-3]
    image_name = image_path_split[-1]

    # joint 2d visualize
    image_i_bgr = cv2.imread(data_i['img_metas'].data['image_file'])

    # visualize the 2d heatmap overlayed on the image
    for i in range(heatmap_2d.shape[0]):
        from mmpose.data.keypoints_mapping.renderpeople import render_people_orginal_joint_names
        heatmap_i = heatmap_2d[i]
        print(heatmap_2d_visible)
        heatmap_i = (heatmap_i + 1) if heatmap_2d_visible[i] < 0.00000001 else heatmap_i
        heatmap_i = heatmap_i * 255
        heatmap_i = heatmap_i.astype(np.uint8)
        heatmap_i = cv2.applyColorMap(heatmap_i, cv2.COLORMAP_JET)
        heatmap_i = cv2.resize(heatmap_i, (1024, 1024))
        # overlay the heatmap on the image
        image_i_bgr_overlay = image_i_bgr.copy()[:, 128: -128, :]
        image_i_bgr_overlay = cv2.addWeighted(image_i_bgr_overlay, 0.5, heatmap_i, 0.5, 0)
        # save the image
        heatmap_i_output_path = os.path.join(heatmap_save_dir, f'{render_people_orginal_joint_names[i]}.jpg')
        cv2.imwrite(heatmap_i_output_path, image_i_bgr_overlay)

    joint_2d = data_i['img_metas'].data['joints_3d'][:, :2] * 4
    joint_2d[:, 0] += 128
    # image_i_bgr = draw_joints(joint_2d, image_i_bgr)
    image_i_bgr = draw_keypoints(joint_2d, image_i_bgr, radius=1)
    joint_2d_output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(joint_2d_output_path, image_i_bgr)

    # hand 2d seg visualize
    left_hand_img = data_i['img_metas'].data['left_hand_img']
    right_hand_img = data_i['img_metas'].data['right_hand_img']

    left_hand_img_vis = deepcopy(left_hand_img)
    right_hand_img_vis = deepcopy(right_hand_img)

    left_hand_img_vis = left_hand_img_vis.cpu().numpy()
    right_hand_img_vis = right_hand_img_vis.cpu().numpy()
    left_hand_img_vis = np.transpose(left_hand_img_vis, (1, 2, 0)) * 255
    right_hand_img_vis = np.transpose(right_hand_img_vis, (1, 2, 0)) * 255
    left_hand_img_vis = left_hand_img_vis.astype(np.uint8)
    right_hand_img_vis = right_hand_img_vis.astype(np.uint8)

    left_hand_img_output_path = os.path.join(output_dir, f'left_hand_img_{image_id}.jpg')
    right_hand_img_output_path = os.path.join(output_dir, f'right_hand_img_{image_id}.jpg')
    cv2.imwrite(left_hand_img_output_path, left_hand_img_vis)
    cv2.imwrite(right_hand_img_output_path, right_hand_img_vis)

    left_hand_img = torch.unsqueeze(left_hand_img, dim=0).cuda()
    right_hand_img = torch.unsqueeze(right_hand_img, dim=0).cuda()
    return left_hand_img, right_hand_img, data_i['img_metas']


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

def run_hand_pose_model(left_hand_img, right_hand_img, img_metas, img_id):
    model_path = '/CT/EgoMocap/work/EgocentricHand/Hand4Whole_RELEASE/demo/hand/snapshot_12.pth.tar'
    assert os.path.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))

    hand_pose_model = dict(
        type='EgoHandPose',
        pretrained=model_path,
    )

    hand_pose_model = build_posenet(hand_pose_model)

    hand_pose_model.cuda()
    hand_pose_model.eval()

    output = hand_pose_model(left_hand_img, right_hand_img, img_metas=img_metas, return_loss=False)
    print(output.keys())

    left_hand_preds = output['left_hands_preds']
    right_hand_preds = output['right_hands_preds']
    print(left_hand_preds.keys())
    print(right_hand_preds.keys())

    # visualize and save mesh

    left_hand_verts = left_hand_preds['mano_mesh_cam']
    left_hand_verts = left_hand_verts.cpu().detach().numpy()[0]
    left_hand_verts[:, 0] *= -1
    right_hand_verts = right_hand_preds['mano_mesh_cam']
    right_hand_verts = right_hand_verts.cpu().detach().numpy()[0]

    left_hand_mesh_out_path = os.path.join(output_dir, f'left_hand_mesh_{img_id}.obj')
    save_obj(left_hand_verts, mano.face['left'], left_hand_mesh_out_path)

    right_hand_mesh_out_path = os.path.join(output_dir, f'right_hand_mesh_{img_id}.obj')
    save_obj(right_hand_verts, mano.face['right'], right_hand_mesh_out_path)



if __name__ == '__main__':
    output_dir = os.path.join('/CT/EgoMocap/work/EgocentricFullBody/vis_results/hands_pose')
    os.makedirs(output_dir, exist_ok=True)

    img_id = 9900
    left_hand_img, right_hand_img, img_metas = test_studio_with_hand_dataset(img_id)

    run_hand_pose_model(left_hand_img, right_hand_img, img_metas, img_id)
