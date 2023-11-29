import copy
import os
import pickle

import cv2
import numpy as np
import open3d
from torch.utils.data import Dataset

from mmpose.datasets.builder import DATASETS
from mmpose.datasets.pipelines import Compose


@DATASETS.register_module()
class EgoPWFinetuneDataset(Dataset):
    """
    {'image_path': image_path_list[i],
                  'global_pose': global_optimized_pose_seq[i],
                  'gt_pose': gt_pose_seq[i],
                  'joints_2d': joints_2d_list[i],
                  'depth': depth_list[i]}
    """

    def __init__(self,
                 root_path,
                 data_cfg,
                 pipeline,
                 test_mode=False):
        """

        :param root_data_path:
        :param is_train:
        :param is_zoom:
        :param local_machine:
        :param use_estimated_pose: use estimated pose as the pseudo ground truth, default: False
        """
        self.root_path = root_path
        self.data_cfg = copy.deepcopy(data_cfg)
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.ann_info = {}

        self.load_config(self.data_cfg)

        self.data_info = self.load_annotations()
        self.pipeline = Compose(pipeline)

    def load_annotations(self):
        # get data
        data = []
        if self.test_mode is False:
            identity_name_list = ['ayush', 'ayush_new', 'binchen', 'chao', 'chao_new',
                                  'kripa', 'kripa_new', 'lingjie', 'lingjie_new', 'mohamed', 'soshi_new']
        else:
            identity_name_list = ['ayush_new', 'kripa_new', 'lingjie_new', 'soshi_new']
        for identity_name in identity_name_list:
            identity_path = os.path.join(self.root_path, identity_name)
            data.extend(self.get_real_identity_data(identity_path))
        return data

    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """
        self.ann_info = copy.deepcopy(data_cfg)
        self.ann_info['image_size'] = np.asarray(self.ann_info['image_size'])

    def get_real_data_single_seq(self, seq_dir):
        pkl_path = os.path.join(seq_dir, 'pseudo_gt.pkl')
        with open(pkl_path, 'rb') as f:
            seq_data = pickle.load(f)
        return seq_data

    def get_real_identity_data(self, identity_path):
        identity_data = []
        for seq_name in os.listdir(identity_path):
            seq_dir = os.path.join(identity_path, seq_name)
            # if 'rountunda' in seq_dir:
            #     continue
            if os.path.isdir(seq_dir):
                seq_data = self.get_real_data_single_seq(seq_dir)
                identity_data.extend(seq_data)

        return identity_data


    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        results = copy.deepcopy(self.prepare_data(idx))
        results['ann_info'] = self.ann_info
        try:
            results = self.pipeline(results)
        except Exception as e:
            print(e)
            print(results['image_file'])
            return self.__getitem__(idx + 1)
        return results
    def prepare_data(self, index):
        data_i = self.data_info[index]
        image_path = data_i['image_path']
        gt_pose = data_i['optimized_local_pose']
        return dict(image_file=image_path,
                    keypoints_3d=gt_pose,
                    keypoints_3d_visible=np.ones([15], dtype=np.float32),
                    )

    def evaluate(self, outputs, res_folder, metric=None, logger=None):
        # just save the outputs
        save_path = os.path.join(res_folder, f'outputs.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(outputs, f)
        evaluation_results = {'mpjpe': 0}
        return evaluation_results


if __name__ == '__main__':
    dataset = EgoPWFinetuneDataset(root_data_path=r'X:\Mo2Cap2Plus1\static00\ExternalEgo\External_camera_all',
                                   local_machine=True,
                                   use_estimated_pose=False,
                                   with_segmentation=True,
                                   camera_model='../utils/fisheye/fisheye.calibration.json')
    print(len(dataset))
    # 3150
    img_torch, image_path, heatmap, joints, depth, seg, gt_pose = dataset[68050]

    joints = joints.numpy()
    depth = depth.numpy()

    joints = joints * 4
    joints = np.add(joints, [128, 0])

    print(image_path)

    lines = [(0, 1, 'right'), (0, 4, 'left'), (1, 2, 'right'), (2, 3, 'right'), (4, 5, 'left'), (5, 6, 'left'),
             (1, 7, 'right'), (4, 11, 'left'), (7, 8, 'right'), (8, 9, 'right'), (9, 10, 'right'),
             (11, 12, 'left'), (12, 13, 'left'), (13, 14, 'left'), (7, 11, 'left')]

    img = cv2.imread(image_path)
    img = draw_joints(joints, img)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    # show segmentation result
    segmentation = seg.numpy()
    segmentation = cv2.resize(segmentation, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    segmentated_result = np.empty_like(img)
    for i in range(3):
        segmentated_result[:, :, i] = img[:, :, i] * segmentation
    cv2.imshow('seg', segmentated_result)
    cv2.waitKey(0)

    from utils_proj.skeleton import Skeleton

    skeleton_model = Skeleton(
        calibration_path=r'X:\Mo2Cap2Plus\work\BodyPoseOptimization\utils\fisheye\fisheye.calibration.json')

    heatmap = heatmap.numpy()
    resized_heatmap = np.empty((15, 1024, 1280))
    for i in range(len(heatmap)):
        resized_heatmap_i = cv2.resize(heatmap[i], dsize=(1024, 1024))
        resized_heatmap[i] = np.pad(resized_heatmap_i, ((0, 0), (128, 128)), mode='edge')

    skeleton_model.set_skeleton(resized_heatmap, depth)
    mesh = skeleton_model.skeleton_mesh
    open3d.visualization.draw_geometries([mesh])
