# pose visualizer
# 1. read and generate 3D skeleton from heat map and depth
# 2. convert 3D skeleton to skeleton mesh
from mmpose.utils.fisheye_camera.FishEyeEquisolid import FishEyeCameraEquisolid
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated
import numpy as np
import open3d
from mmpose.utils.visualization.pose_visualization_utils import get_cylinder, get_sphere
from scipy.io import loadmat
import cv2
import os
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter1d


class HandSkeleton:

    
    def __init__(self, hand_model):
        if hand_model == 'mano':
            from mmpose.data.keypoints_mapping.mano import mano_joint_names, mano_skeleton
            self.joint_sequence = mano_joint_names
            self.skeleton = mano_skeleton
    
    def joints_2_mesh(self, joints_3d, joint_color=(0.1, 0.1, 0.7), bone_color=(0.1, 0.9, 0.1)):
        final_mesh = open3d.geometry.TriangleMesh()
        for i in range(len(joints_3d)):
            keypoint_mesh = get_sphere(position=joints_3d[i], radius=0.008, color=joint_color)
            final_mesh = final_mesh + keypoint_mesh

        for line in self.skeleton:
            line_start_i = line[0]
            line_end_i = line[1]

            start_point = joints_3d[line_start_i]
            end_point = joints_3d[line_end_i]

            line_mesh = get_cylinder(start_point, end_point, radius=0.0006, color=bone_color)
            final_mesh += line_mesh
        return final_mesh

    

