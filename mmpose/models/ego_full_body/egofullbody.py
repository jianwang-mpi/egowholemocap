#  Copyright Jian Wang @ MPI-INF (c) 2023.

import warnings
from copy import deepcopy

import numpy as np
import torch
from mmcv.runner.checkpoint import load_checkpoint

from mmpose.models.detectors.base import BasePose
from .. import builder
from ..builder import POSENETS
from ...core.evaluation.top_down_eval import _get_softargmax_preds
from ...datasets.pipelines import Compose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class EgocentricFullBodyPose(BasePose):
    """
    egocentric 3d pose estimation for hand and body
    """

    def __init__(self,
                 body_pose_dict,
                 hand_detection_dict,
                 hand_pose_estimation_dict,
                 hand_process_pipeline,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained_body_pose=None,
                 pretrained_hand_detection=None,
                 pretrained_hand_pose_estimation=None,

                 ):
        super(EgocentricFullBodyPose, self).__init__()

        self.body_pose_module = builder.build_posenet(body_pose_dict)
        self.hand_detection_module = builder.build_posenet(hand_detection_dict)
        self.hand_pose_estimation_module = builder.build_posenet(hand_pose_estimation_dict)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fisheye_camera_path = self.test_cfg['fisheye_camera_path']
        if hand_process_pipeline is None:
            hand_process_pipeline = [
                dict(type='CropHandImageFisheye', fisheye_camera_path=self.fisheye_camera_path,
                     input_img_h=1024, input_img_w=1280,
                     crop_img_size=256, enlarge_scale=1.3),
                dict(type='RGB2BGRHand'),
                dict(type='ToTensorHand'),
            ]
        self.hand_process_pipeline = Compose(hand_process_pipeline)

        load_checkpoint(self.body_pose_module, pretrained_body_pose, map_location='cpu')
        load_checkpoint(self.hand_detection_module, pretrained_hand_detection, map_location='cpu')
        if pretrained_hand_pose_estimation is not None:
            load_checkpoint(self.hand_pose_estimation_module, pretrained_hand_pose_estimation, map_location='cpu')
        else:
            print('do not load checkpoint for hand pose estimation module')

    def forward_train(self, img, img_original, keypoints_body, keypoints_body_visible,
                      keypoints_left_hand, keypoints_left_hand_visible,
                      keypoints_right_hand, keypoints_right_hand_visible, img_metas, **kwargs):
        pass

    def forward_test(self, img, img_original, img_metas, left_hand_img=None, right_hand_img=None,
                     left_hand_transform=None, right_hand_transform=None, **kwargs):
        # estimate body pose
        body_pose_results = self.body_pose_module(img, img_metas=img_metas, return_loss=False, **kwargs)
        use_hand_detection_module = False
        if left_hand_img is None:
            # print('use hand detection module')
            use_hand_detection_module = True
            hand_2d_pose_heatmaps = self.hand_detection_module(img, img_metas=img_metas, return_loss=False, **kwargs)
            crop_output_dict = self.process_hand(img, img_original, hand_2d_pose_heatmaps['output_heatmap'])
            left_hand_img = torch.from_numpy(crop_output_dict['left_hand_img']).float().to(img.device)
            right_hand_img = torch.from_numpy(crop_output_dict['right_hand_img']).float().to(img.device)
            left_hand_transform = torch.from_numpy(crop_output_dict['left_hand_transform']).float().to(img.device)
            right_hand_transform = torch.from_numpy(crop_output_dict['right_hand_transform']).float().to(img.device)

        result_dict = {}
        result_dict['img_metas'] = img_metas
        result_dict['body_pose_results'] = body_pose_results
        result_dict['keypoints_pred'] = body_pose_results['keypoints_pred']
        # result_dict['left_hand_img'] = left_hand_img
        # result_dict['right_hand_img'] = right_hand_img
        result_dict['left_hand_transform'] = left_hand_transform
        result_dict['right_hand_transform'] = right_hand_transform
        if use_hand_detection_module:
            result_dict['left_hand_keypoints_2d'] = crop_output_dict['left_hand_keypoints_2d']
            result_dict['right_hand_keypoints_2d'] = crop_output_dict['right_hand_keypoints_2d']

        hand_pose_estimation_result = self.hand_pose_estimation_module(left_hand_img, right_hand_img,
                                                                       left_hand_transform, right_hand_transform,
                                                                       img_metas=img_metas,
                                                                       return_loss=False, **kwargs)
        result_dict['left_hands_preds'] = hand_pose_estimation_result['left_hands_preds']
        result_dict['right_hands_preds'] = hand_pose_estimation_result['right_hands_preds']
        return result_dict

    def forward(self, img, img_original, left_hand_img=None, right_hand_img=None,
                left_hand_transform=None, right_hand_transform=None,
                keypoints_body=None, keypoints_body_visible=None,
                keypoints_left_hand=None, keypoints_left_hand_visible=None,
                keypoints_right_hand=None, keypoints_right_hand_visible=None,
                img_metas=None, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_original, keypoints_body, keypoints_body_visible, keypoints_left_hand,
                                      keypoints_left_hand_visible, keypoints_right_hand,
                                      keypoints_right_hand_visible, img_metas, **kwargs)
        return self.forward_test(
            img, img_original, img_metas=img_metas, left_hand_img=left_hand_img, right_hand_img=right_hand_img,
            left_hand_transform=left_hand_transform, right_hand_transform=right_hand_transform,
            **kwargs)

    def process_hand(self, img, img_original, hand_2d_heatmap):
        batch_size_img, _, img_height, img_width = img.shape
        batch_size_img_orginal, img_height_original, img_width_original, __ = img_original.shape
        batch_size_heatmap, hand_joint_num, img_height_heatmap, img_width_heatmap = hand_2d_heatmap.shape
        assert img_height == 256 and img_width == 256 and img_height_original == 1024 and img_width_original == 1280
        assert batch_size_img == batch_size_img_orginal and batch_size_img == batch_size_heatmap
        assert img_height_heatmap == 64 and img_width_heatmap == 64
        assert hand_joint_num == 21 * 2

        img_original = img_original.detach().cpu().numpy()

        # get 2d hand pose from the heatmap
        hand_2d_heatmap_np = hand_2d_heatmap
        hand_2d_pose, _ = _get_softargmax_preds(hand_2d_heatmap_np)
        hand_2d_pose_original = hand_2d_pose * (img_height_original // img_height_heatmap)
        hand_2d_pose_original[:, :, 0] += 128

        left_hand_2d_pose = hand_2d_pose_original[:, :21]
        right_hand_2d_pose = hand_2d_pose_original[:, 21:]
        assert len(img_original) == len(left_hand_2d_pose) == len(right_hand_2d_pose)
        result_list = []
        for i in range(len(img_original)):
            input_dict = {
                'img': img_original[i],
                'left_hand_keypoints_2d': left_hand_2d_pose[i],
                'right_hand_keypoints_2d': right_hand_2d_pose[i],
            }
            output_dict_item = self.hand_process_pipeline(input_dict)
            result_list.append(output_dict_item)
        output_dict = dict()
        for key in result_list[0].keys():
            output_dict[key] = []
            for i in range(len(result_list)):
                output_dict[key].append(result_list[i][key])
            output_dict[key] = np.stack(output_dict[key], axis=0)
        output_dict['left_hand_keypoints_2d'] = deepcopy(left_hand_2d_pose)
        output_dict['right_hand_keypoints_2d'] = deepcopy(right_hand_2d_pose)
        return output_dict

    def show_result(self, **kwargs):
        pass
