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


class Skeleton:
    heatmap_sequence = ["Neck", "Right_shoulder", "Right_elbow", "Right_wrist", "Left_shoulder", "Left_elbow",
                        "Left_wrist", "Right_hip", "Right_knee", "Right_ankle", "Right_foot", "Left_hip",
                        "Left_knee", "Left_ankle", "Left_foot"]
    lines = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6), (1, 7), (4, 11), (7, 8), (8, 9), (9, 10),
             (11, 12), (12, 13), (13, 14), (7, 11)]
    kinematic_parents = [0, 0, 1, 2, 0, 4, 5, 1, 7, 8, 9, 4, 11, 12, 13]
    
    def __init__(self, calibration_path):
        
        self.skeleton = None
        self.skeleton_mesh = None
        if calibration_path is None:
            print('use FishEyeCameraEquisolid')
            self.camera = FishEyeCameraEquisolid(focal_length=9, sensor_size=32, img_size=(1280, 1024))
        else:
            self.camera = FishEyeCameraCalibrated(calibration_file_path=calibration_path)
    
    def set_skeleton(self, heatmap, depth, bone_length=None):
        heatmap = np.expand_dims(heatmap, axis=0)
        preds, _ = self.get_max_preds(heatmap)
        pred = preds[0]
        
        points_3d = self.camera.camera2world(pred, depth)
        # print('------------------------')
        # print(self.camera.camera2world(np.array([[640, 1000]]), np.array([1])))
        if bone_length is not None:
            points_3d = self._skeleton_resize(points_3d, bone_length)
        return points_3d

    def get_2d_pose_from_heatmap(self, heatmap):
        heatmap = np.expand_dims(heatmap, axis=0)
        preds, _ = self.get_max_preds(heatmap)
        pred = preds[0]
        return pred
    
    def joints_2_mesh(self, joints_3d, joint_color=(0.1, 0.1, 0.7), bone_color=(0.1, 0.9, 0.1)):
        self.skeleton = joints_3d
        self.skeleton_to_mesh(joint_color, bone_color)
        skeleton_mesh = self.skeleton_mesh
        self.skeleton_mesh = None
        self.skeleton = None
        return skeleton_mesh
    
    def joint_list_2_mesh_list(self, joints_3d_list):
        mesh_list = []
        for joints_3d in joints_3d_list:
            mesh_list.append(self.joints_2_mesh(joints_3d))
        return mesh_list
    
    def get_skeleton_mesh(self):
        if self.skeleton_mesh is None:
            raise Exception("Skeleton is not prepared.")
        else:
            return self.skeleton_mesh
    
    def save_skeleton_mesh(self, out_path):
        if self.skeleton_mesh is None:
            raise Exception("Skeleton is not prepared.")
        else:
            open3d.io.write_triangle_mesh(out_path, mesh=self.skeleton_mesh)
    
    def set_skeleton_from_file(self, heatmap_file, depth_file, bone_length_file=None, to_mesh=True):
        # load the average bone length
        if bone_length_file is not None:
            bone_length_mat = loadmat(bone_length_file)
            mean3D = bone_length_mat['mean3D'].T  # convert shape to 15 * 3
            bones_mean = mean3D - mean3D[self.kinematic_parents, :]
            bone_length = np.linalg.norm(bones_mean, axis=1)
        else:
            bone_length = None
        heatmap_mat = loadmat(heatmap_file)
        depth_mat = loadmat(depth_file)
        depth = depth_mat['depth'][0]
        heatmap = heatmap_mat['heatmap']
        heatmap = cv2.resize(heatmap, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
        heatmap = np.pad(heatmap, ((0, 0), (128, 128), (0, 0)), 'constant', constant_values=0)
        heatmap = heatmap.transpose((2, 0, 1))
        return self.set_skeleton(heatmap, depth, bone_length, to_mesh)
    
    def skeleton_resize_seq(self, joint_list, bone_length_file):
        bone_length_mat = loadmat(bone_length_file)
        mean3D = bone_length_mat['mean3D'].T  # convert shape to 15 * 3
        bones_mean = mean3D - mean3D[self.kinematic_parents, :]
        bone_length = np.linalg.norm(bones_mean, axis=1)
        
        for i in range(len(joint_list)):
            joint_list[i] = self._skeleton_resize(joint_list[i], bone_length)
        return joint_list
    
    def skeleton_resize_single(self, joint, bone_length_file):
        bone_length_mat = loadmat(bone_length_file)
        mean3D = bone_length_mat['mean3D'].T  # convert shape to 15 * 3
        bones_mean = mean3D - mean3D[self.kinematic_parents, :]
        bone_length = np.linalg.norm(bones_mean, axis=1)
        
        joint = self._skeleton_resize(joint, bone_length)
        return joint
    
    def skeleton_resize_standard_skeleton(self, joint_input, joint_standard):
        """
        
        :param joint_input: input joint shape: 15 * 3
        :param joint_standard: standard joint shape: 15 * 3
        :return:
        """
        bones_mean = joint_standard - joint_standard[self.kinematic_parents, :]
        bone_length = np.linalg.norm(bones_mean, axis=1) * 1000.
    
        joint = self._skeleton_resize(joint_input, bone_length)
        return joint
    
    def _skeleton_resize(self, points_3d, bone_length):
        # resize the skeleton to the normal size (why we should do that?)
        estimated_bone_vec = points_3d - points_3d[self.kinematic_parents, :]
        estimated_bone_length = np.linalg.norm(estimated_bone_vec, axis=1)
        multi = bone_length[1:] / estimated_bone_length[1:]
        multi = np.concatenate(([0], multi))
        multi = np.stack([multi] * 3, axis=1)
        resized_bones_vec = estimated_bone_vec * multi / 1000
        
        joints_rescaled = points_3d
        for i in range(joints_rescaled.shape[0]):
            joints_rescaled[i, :] = joints_rescaled[self.kinematic_parents[i], :] + resized_bones_vec[i, :]
        return joints_rescaled
    
    def render(self):
        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        open3d.visualization.draw_geometries([self.skeleton_mesh, mesh_frame])
    
    def skeleton_to_mesh(self, joint_color=(0.1, 0.1, 0.7), bone_color=(0.1, 0.9, 0.1)):
        final_mesh = open3d.geometry.TriangleMesh()
        for i in range(len(self.skeleton)):
            keypoint_mesh = get_sphere(position=self.skeleton[i], radius=0.03, color=joint_color)
            final_mesh = final_mesh + keypoint_mesh
        
        for line in self.lines:
            line_start_i = line[0]
            line_end_i = line[1]
            
            start_point = self.skeleton[line_start_i]
            end_point = self.skeleton[line_end_i]
            
            line_mesh = get_cylinder(start_point, end_point, radius=0.0075, color=bone_color)
            final_mesh += line_mesh
        self.skeleton_mesh = final_mesh
        return final_mesh
    
    def smooth(self, pose_sequence, sigma):
        """
        gaussian smooth pose
        :param pose_sequence_2d: pose sequence, input is a list with every element is 15 * 2 body pose
        :param kernel_size: kernel size of guassian smooth
        :return: smoothed 2d pose
        """
        pose_sequence = np.asarray(pose_sequence)
        pose_sequence_result = np.zeros_like(pose_sequence)
        keypoint_num = pose_sequence.shape[1]
        for i in range(keypoint_num):
            pose_sequence_i = pose_sequence[:, i, :]
            pose_sequence_filtered = gaussian_filter1d(pose_sequence_i, sigma, axis=0)
            pose_sequence_result[:, i, :] = pose_sequence_filtered
        return pose_sequence_result
    
    def get_max_preds(self, batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, np.ndarray), \
            'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'
        
        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)
        
        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))
        
        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        
        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)
        
        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)
        
        preds *= pred_mask
        return preds, maxvals


if __name__ == '__main__':
    skeleton = Skeleton(
        calibration_path='/home/wangjian/Develop/egocentricvisualization/pose/fisheye/fisheye.calibration.json')
    data_path = r'/home/wangjian/Develop/egocentricvisualization/data_2'
    heatmap_dir = os.path.join(data_path, 'heatmaps')
    depth_dir = os.path.join(data_path, 'depths')
    out_dir = os.path.join(data_path, 'smooth_skeleton_mesh')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    skeleon_list = []
    out_path_list = []
    for heatmap_name in tqdm(sorted(os.listdir(heatmap_dir))):
        heatmap_path = os.path.join(heatmap_dir, heatmap_name)
        mat_id = heatmap_name
        depth_path = os.path.join(depth_dir, mat_id)
        
        skeleton_array = skeleton.set_skeleton_from_file(heatmap_path,
                                                         depth_path,
                                                         # bone_length_file=r'/home/wangjian/Develop/egocentricvisualization/pose/fisheye/mean3D.mat',
                                                         to_mesh=False)
        out_path = os.path.join(out_dir, mat_id + ".ply")
        skeleon_list.append(skeleton_array)
        out_path_list.append(out_path)
    
    smoothed_skeleton = skeleton.smooth(skeleon_list, sigma=1)
    print("saving to ply")
    for i in tqdm(range(len(smoothed_skeleton))):
        skeleton.skeleton = smoothed_skeleton[i]
        skeleton.skeleton_to_mesh()
        skeleton.save_skeleton_mesh(out_path_list[i])
    
    # skeleton.set_skeleton_from_file(r'X:\Mo2Cap2Plus\static00\Datasets\Mo2Cap2\ego_system_test\sitting\heatmaps\img-04052020001910-937.mat',
    #                                 r'X:\Mo2Cap2Plus\static00\Datasets\Mo2Cap2\ego_system_test\sitting\depths\img-04052020001910-937.mat',
    #                                 # bone_length_file=r'F:\Develop\EgocentricSystemVisualization\pose\fisheye\mean3D.mat')
    #                                 )
    #
    # skeleton.render()
    # print(skeleton.skeleton)
