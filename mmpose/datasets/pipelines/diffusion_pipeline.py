#  Copyright Jian Wang @ MPI-INF (c) 2023.
import pickle
from copy import deepcopy

import numpy as np
import torch

from mmpose.datasets.builder import PIPELINES
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
from mmpose.models.diffusion_mdm.data_loaders.humanml.common.quaternion import qrot_np, qbetween_np, qmul_np, qinv_np, qeuler


@PIPELINES.register_module()
class CalculateEgocentricCameraLocationFromSMPLX:
    def __init__(self, random_camera_rotation=False, random_camera_translation=False):
        super(CalculateEgocentricCameraLocationFromSMPLX, self).__init__()
        self.random_camera_rotation = random_camera_rotation
        self.random_camera_translation = random_camera_translation

    def calculate_camera_location(self, smplx_vertices):
        # smplx_vertices shape: (seq_len, 10475, 3)
        if self.random_camera_rotation:
            back_head_vertex_id = np.random.choice([8961, 8979, 8973], p=[0.9, 0.05, 0.05])
        else:
            back_head_vertex_id = 8961
        midpoint_eyes = smplx_vertices[:, 9004]
        left_ear_to_right_ear_vector = smplx_vertices[:, 1316] - smplx_vertices[:, 451]
        X = left_ear_to_right_ear_vector / np.linalg.norm(left_ear_to_right_ear_vector, axis=1)[:, None]
        front_head_to_back_head_vector = smplx_vertices[:, back_head_vertex_id] - midpoint_eyes
        Y = front_head_to_back_head_vector / np.linalg.norm(front_head_to_back_head_vector, axis=1)[:, None]
        Y = Y - np.einsum('ij,ij->i', X, Y)[:, None] * X
        # make X vertical to Y, since X and Y are normalized
        Y = Y / np.linalg.norm(Y, axis=1)[:, None]
        Z = np.cross(X, Y, axis=1)
        camera_rotation = np.stack([X, Y, Z], axis=1)
        camera_rotation = np.transpose(camera_rotation, (0, 2, 1))

        if self.random_camera_translation:
            camera_transl_y = np.random.uniform(0.9, 1.1) * 0.157
            camera_transl_z = np.random.uniform(0.9, 1.1) * 0.01
        else:
            camera_transl_y = 0.255
            camera_transl_z = 0.01
        camera_transl = midpoint_eyes - Y * camera_transl_y + Z * camera_transl_z

        return camera_rotation, camera_transl

    def __call__(self, results: dict) -> dict:
        smplx_output = results['smplx_output']
        smplx_vertices = smplx_output.vertices.cpu().numpy()

        camera_rotation_list, camera_transl_list = self.calculate_camera_location(smplx_vertices)
        results['ego_camera_rot'] = camera_rotation_list
        results['ego_camera_transl'] = camera_transl_list

        # build transform matrix from rotation and translation
        ego_camera_transform = np.zeros((len(camera_rotation_list), 4, 4))
        ego_camera_transform[:, :3, :3] = camera_rotation_list
        ego_camera_transform[:, :3, 3] = camera_transl_list
        ego_camera_transform[:, 3, 3] = 1
        results['ego_camera_transform'] = ego_camera_transform

        return results



@PIPELINES.register_module()
class ConvertSMPLXOutputToEgocentricCameraLocation:
    def __init__(self):
        super(ConvertSMPLXOutputToEgocentricCameraLocation, self).__init__()

    def __call__(self, results: dict) -> dict:
        ego_camera_transform_seq = results['ego_camera_transform']

        smplx_output = results['smplx_output']

        smplx_joints = smplx_output.joints.detach()
        smplx_vertices = smplx_output.vertices.detach()
        ego_camera_transform_seq = torch.from_numpy(ego_camera_transform_seq).float().to(smplx_joints.device)
        matrix = torch.linalg.inv(ego_camera_transform_seq)

        # convert the smplx joints to local coordinate system
        seq_len, joint_num, _ = smplx_joints.shape
        _, vertices_num, _ = smplx_vertices.shape
        assert _ == 3

        smplx_joints_homo = torch.concatenate([smplx_joints, torch.ones((seq_len, joint_num, 1))], dim=2)
        ego_smplx_joints = torch.bmm(matrix, smplx_joints_homo.permute(0, 2, 1)).permute(0, 2, 1)
        ego_smplx_joints = ego_smplx_joints[:, :, :3]
        results['ego_smplx_joints'] = ego_smplx_joints

        smplx_vertices_homo = torch.concatenate([smplx_vertices, torch.ones((seq_len, vertices_num, 1))], dim=2)
        ego_smplx_vertices = torch.bmm(matrix, smplx_vertices_homo.permute(0, 2, 1)).permute(0, 2, 1)
        ego_smplx_vertices = ego_smplx_vertices[:, :, :3]
        results['ego_smplx_vertices'] = ego_smplx_vertices
        return results


@PIPELINES.register_module()
class SplitEgoHandMotion:
    def __init__(self):
        super(SplitEgoHandMotion, self).__init__()

        self.left_hand_dst_idxs, self.left_hand_model_idxs = dset_to_body_model(dset='mano_left', model_type='smplx')
        self.right_hand_dst_idxs, self.right_hand_model_idxs = dset_to_body_model(dset='mano_right',
                                                                                  model_type='smplx')

    def __call__(self, results: dict) -> dict:
        # split egocentric hand joints
        ego_smplx_joints = results['ego_smplx_joints']
        seq_len = ego_smplx_joints.shape[0]

        left_hand_keypoints_3d = torch.zeros([seq_len, 21, 3]).float()
        right_hand_keypoints_3d = torch.zeros([seq_len, 21, 3]).float()
        left_hand_keypoints_3d[:, self.left_hand_dst_idxs] = ego_smplx_joints[:, self.left_hand_model_idxs]
        right_hand_keypoints_3d[:, self.right_hand_dst_idxs] = ego_smplx_joints[:, self.right_hand_model_idxs]

        results['left_hand_keypoints_3d'] = left_hand_keypoints_3d
        results['right_hand_keypoints_3d'] = right_hand_keypoints_3d
        return results


@PIPELINES.register_module()
class PreProcessHandMotion:
    def __init__(self, mean_std_path=None, normalize=True, use_20_joints=False, align_hand_root=False):
        self.normalize = normalize
        self.use_20_joints = use_20_joints
        self.align_hand_root = align_hand_root
        if self.normalize:
            with open(mean_std_path, 'rb') as f:
                mean_std = pickle.load(f)

            self.left_hand_mean = torch.from_numpy(mean_std['left_hand_mean']).float()
            self.left_hand_std = torch.from_numpy(mean_std['left_hand_std']).float()

            self.right_hand_mean = torch.from_numpy(mean_std['right_hand_mean']).float()
            self.right_hand_std = torch.from_numpy(mean_std['right_hand_std']).float()

    def process_hand_keypoints(self, hand_keypoints, align_hand_root=False):
        # hand_keypoints shape: (seq_len, 21, 3)
        if align_hand_root:
            hand_root_pos = deepcopy(hand_keypoints[:, 0:1])
            hand_keypoints = hand_keypoints - hand_root_pos
        else:
            hand_keypoints[:, 1:] -= hand_keypoints[:, 0:1]
        return hand_keypoints

    def __call__(self, results: dict) -> dict:
        # preprocess the egocentric hand joints
        # process the egocentric hand joints in egocentric space to the input representation of the diffusion model
        left_hand_keypoints_3d = deepcopy(results['left_hand_keypoints_3d'])
        right_hand_keypoints_3d = deepcopy(results['right_hand_keypoints_3d'])

        left_hand_keypoints_3d = self.process_hand_keypoints(left_hand_keypoints_3d, self.align_hand_root)
        right_hand_keypoints_3d = self.process_hand_keypoints(right_hand_keypoints_3d, self.align_hand_root)

        results['processed_left_hand_keypoints_3d'] = left_hand_keypoints_3d
        results['processed_right_hand_keypoints_3d'] = right_hand_keypoints_3d

        left_hand_feature = left_hand_keypoints_3d.reshape(left_hand_keypoints_3d.shape[0], 21 * 3)
        right_hand_feature = right_hand_keypoints_3d.reshape(right_hand_keypoints_3d.shape[0], 21 * 3)
        # normalize
        if self.normalize:
            left_hand_feature = (left_hand_feature - self.left_hand_mean) / self.left_hand_std
            right_hand_feature = (right_hand_feature - self.right_hand_mean) / self.right_hand_std
        if self.use_20_joints:
            left_hand_feature = left_hand_feature[:, 3:]
            right_hand_feature = right_hand_feature[:, 3:]
        results['left_hand_features'] = left_hand_feature
        results['right_hand_features'] = right_hand_feature

        return results


@PIPELINES.register_module()
class PreProcessMo2Cap2BodyMotion:
    def __init__(self, mean_std_path=None, normalize=True):
        self.normalize = normalize
        if self.normalize:
            with open(mean_std_path, 'rb') as f:
                mean_std = pickle.load(f)

            self.mean = torch.from_numpy(mean_std['mo2cap2_body_mean']).float()
            self.std = torch.from_numpy(mean_std['mo2cap2_body_std']).float()

    def __call__(self, results: dict) -> dict:
        mo2cap2_keypoints_3d = deepcopy(results['mo2cap2_keypoints_3d'])

        mo2cap2_body_features = mo2cap2_keypoints_3d.reshape(mo2cap2_keypoints_3d.shape[0], 15 * 3)

        # normalize
        if self.normalize:
            mo2cap2_body_features = (mo2cap2_body_features - self.mean) / self.std
        results['mo2cap2_body_features'] = mo2cap2_body_features

        return results

@PIPELINES.register_module()
class PreProcessRootMotion:
    def __init__(self, mean_std_path=None, normalize=True):
        self.normalize = normalize
        if self.normalize:
            with open(mean_std_path, 'rb') as f:
                mean_std = pickle.load(f)

            self.mean = torch.from_numpy(mean_std['root_features_mean']).float()
            self.std = torch.from_numpy(mean_std['root_features_std']).float()

    def __call__(self, results: dict) -> dict:
        local_root_velocity = results['local_root_velocity']
        assert local_root_velocity.shape[-1] == 3
        assert local_root_velocity[-1, 1] == 0
        local_root_velocity = local_root_velocity[:, [0, 2]]
        local_root_rotation_velocity = results['local_root_rotation_velocity_y']
        assert len(local_root_rotation_velocity.shape) == 1
        local_root_rotation_velocity = local_root_rotation_velocity[:, None]
        root_features = torch.cat([local_root_velocity, local_root_rotation_velocity], dim=-1)
        # normalize
        if self.normalize:
            root_features = (root_features - self.mean) / self.std
        results['root_features'] = root_features
        return results


@PIPELINES.register_module()
class EgoFeaturesNormalize:
    def __init__(self, mean_std_path=None, normalize_name_list=None, to_tensor=False):
        self.normalize_name_list = normalize_name_list
        self.to_tensor = to_tensor
        with open(mean_std_path, 'rb') as f:
            self.mean_std = pickle.load(f)

    def __call__(self, results: dict) -> dict:
        for name in self.normalize_name_list:
            mean = self.mean_std[f'{name}_mean'] if f'{name}_mean' in self.mean_std.keys() else self.mean_std[f'{name[:-9]}_mean']
            std = self.mean_std[f'{name}_std'] if f'{name}_std' in self.mean_std.keys() else self.mean_std[f'{name[:-9]}_std']
            results[name] = (results[name] - mean) / std
            if self.to_tensor:
                results[name] = torch.from_numpy(results[name]).float()
        return results

@PIPELINES.register_module()
class AlignGlobalSMPLXJoints:
    def __init__(self, align_every_joint=False,
                 feet_threshold=0.002, use_default_floor_height=False):
        self.align_every_joint = align_every_joint
        self.feet_threshold = feet_threshold
        self.default_floor_height = use_default_floor_height
        self.hip_ids = [2, 1]
        self.shoulder_ids = [17, 16]
        self.feet_id_r, self.feet_id_l = [8, 11], [7, 10]

    def align_joint_sequence_origin(self, smplx_joints):
        if self.default_floor_height is False:
            floor_height = smplx_joints.min(axis=0).min(axis=0)[1]
            smplx_joints[:, :, 1] -= floor_height
        root_pos_init = smplx_joints[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        smplx_joints = smplx_joints - root_pose_init_xz
        r_hip, l_hip = self.hip_ids
        sdr_r, sdr_l = self.shoulder_ids
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
        target = np.array([[0, 0, 1]])
        root_quat_init = qbetween_np(forward_init, target)
        root_quat_init = np.ones(smplx_joints.shape[:-1] + (4,)) * root_quat_init
        aligned_smplx_joints = qrot_np(root_quat_init, smplx_joints)
        return aligned_smplx_joints


    def get_joint_sequence_global_rotation(self, joint_sequence):
        r_hip, l_hip = self.hip_ids
        sdr_r, sdr_l = self.shoulder_ids
        across1 = joint_sequence[:, r_hip] - joint_sequence[:, l_hip]
        across2 = joint_sequence[:, sdr_r] - joint_sequence[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[:, np.newaxis]

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)
        return root_quat

    def align_every_joints(self, smplx_joints):
        # align every joint to the global coordinate system
        # smplx_joints shape: (seq_len, 127, 3)
        seq_len, joint_num, _ = smplx_joints.shape
        assert _ == 3
        root_quat = self.get_joint_sequence_global_rotation(smplx_joints)

        smplx_joints[..., 0] -= smplx_joints[:, 0:1, 0]
        smplx_joints[..., 2] -= smplx_joints[:, 0:1, 2]
        '''All pose face Z+'''
        smplx_joints = qrot_np(np.repeat(root_quat[:, None], smplx_joints.shape[1], axis=1), smplx_joints)
        return smplx_joints


    def __call__(self, results: dict) -> dict:
        global_smplx_joints = results['global_smplx_joints']
        if torch.is_tensor(global_smplx_joints):
            global_smplx_joints = global_smplx_joints.cpu().numpy()
        seq_len = global_smplx_joints.shape[0]
        # align initial joint
        aligned_smplx_joints = self.align_joint_sequence_origin(global_smplx_joints)
        if self.align_every_joint:
            aligned_smplx_joints = self.align_every_joints(aligned_smplx_joints)
        results['aligned_smplx_joints'] = torch.from_numpy(aligned_smplx_joints).float()
        return results



@PIPELINES.register_module()
class AlignAllGlobalSMPLXJointsWithGlobalInfo:
    '''
    return all information from the alignment process, useful for recovering back the global smplx joints
    '''
    def __init__(self, feet_threshold=0.002, use_default_floor_height=False):
        self.feet_threshold = feet_threshold
        self.default_floor_height = use_default_floor_height
        self.hip_ids = [2, 1]
        self.shoulder_ids = [17, 16]
        self.feet_id_r, self.feet_id_l = [8, 11], [7, 10]

    def get_joint_sequence_global_rotation(self, joint_sequence):
        r_hip, l_hip = self.hip_ids
        sdr_r, sdr_l = self.shoulder_ids
        across1 = joint_sequence[:, r_hip] - joint_sequence[:, l_hip]
        across2 = joint_sequence[:, sdr_r] - joint_sequence[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[:, np.newaxis]

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)
        return root_quat

    def align_joint_sequence_origin(self, global_smplx_joints):
        smplx_joints = deepcopy(global_smplx_joints)
        if self.default_floor_height is False:
            floor_height = smplx_joints.min(axis=0).min(axis=0)[1]
            smplx_joints[:, :, 1] -= floor_height
        root_pos_init = smplx_joints[0]
        root_pose_init_xz = -deepcopy(root_pos_init[0] * np.array([1, 0, 1]))
        smplx_joints = smplx_joints + root_pose_init_xz
        r_hip, l_hip = self.hip_ids
        sdr_r, sdr_l = self.shoulder_ids
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
        target = np.array([[0, 0, 1]])
        root_quat_init = qbetween_np(forward_init, target)
        root_quat_init_ret = deepcopy(root_quat_init)
        root_quat_init = np.ones(smplx_joints.shape[:-1] + (4,)) * root_quat_init
        init_aligned_smplx_joints = qrot_np(root_quat_init, smplx_joints)
        return init_aligned_smplx_joints, root_pose_init_xz, root_quat_init_ret

    def align_every_joints(self, global_smplx_joints):
        # align every joint to the global coordinate system
        # smplx_joints shape: (seq_len, 127, 3)
        smplx_joints = deepcopy(global_smplx_joints)
        seq_len, joint_num, _ = smplx_joints.shape
        assert _ == 3
        root_quat = self.get_joint_sequence_global_rotation(smplx_joints)

        smplx_root_trans = deepcopy(smplx_joints[:, 0:1])
        if self.default_floor_height is True:
            smplx_root_trans[:, :, 1] *= 0  # do not move the root in y axis
        else:
            floor_height = smplx_joints.min(axis=0).min(axis=0)[1]
            smplx_root_trans[:, :, 1] = floor_height
        smplx_root_trans *= -1
        # smplx_joints[..., 0] -= smplx_joints[:, 0:1, 0]
        # smplx_joints[..., 2] -= smplx_joints[:, 0:1, 2]
        smplx_joints += smplx_root_trans
        '''All pose face Z+'''
        smplx_joints = qrot_np(np.repeat(root_quat[:, None], smplx_joints.shape[1], axis=1), smplx_joints)
        return smplx_joints, smplx_root_trans, root_quat

    def calculate_local_rot_trans_velocity(self, global_smplx_joints, root_rot_quat):
        smplx_joints = deepcopy(global_smplx_joints)
        rot_quat = deepcopy(root_rot_quat)
        global_velocity = smplx_joints[1:] - smplx_joints[:-1]
        # global velocity to local velocity
        local_joints_velocity = qrot_np(np.repeat(rot_quat[:-1, None], global_velocity.shape[1], axis=1), global_velocity)
        local_rotation_velocity = qmul_np(rot_quat[1:], qinv_np(rot_quat[:-1]))
        return local_joints_velocity, local_rotation_velocity

    def __call__(self, results: dict) -> dict:
        global_smplx_joints = results['global_smplx_joints']
        if torch.is_tensor(global_smplx_joints):
            global_smplx_joints = global_smplx_joints.cpu().numpy()
        # align initial joint
        init_aligned_smplx_joints, root_init_xz, root_quat_init = self.align_joint_sequence_origin(global_smplx_joints)

        aligned_smplx_joints, root_trans, root_rot_quat = self.align_every_joints(init_aligned_smplx_joints)
        local_joints_velocity, local_root_rotation_velocity = self.calculate_local_rot_trans_velocity(
            init_aligned_smplx_joints, root_rot_quat
        )
        local_root_velocity = deepcopy(local_joints_velocity[:, 0])
        local_root_velocity[:, 1] = 0
        results['aligned_smplx_joints'] = torch.from_numpy(aligned_smplx_joints).float()
        results['root_trans_init_xz'] = torch.from_numpy(root_init_xz).float()
        results['root_rot_quat_init'] = torch.from_numpy(root_quat_init).float()
        results['root_trans_xz'] = torch.from_numpy(root_trans).float()
        results['root_rot_quat'] = torch.from_numpy(root_rot_quat).float()
        results['local_root_velocity'] = torch.from_numpy(local_root_velocity).float()
        results['local_joints_velocity'] = torch.from_numpy(local_joints_velocity).float()
        results['local_root_rotation_velocity'] = torch.from_numpy(local_root_rotation_velocity).float()
        results['local_root_rotation_velocity_y'] = qeuler(results['local_root_rotation_velocity'],
                                                           'xyz', deg=False)[:, 1]
        return results


@PIPELINES.register_module()
class AlignAllGlobalSMPLXJointsWithInfo:
    '''
    return all information from the alignment process, useful for recovering back the global smplx joints
    '''
    def __init__(self, feet_threshold=0.002, use_default_floor_height=False):
        self.feet_threshold = feet_threshold
        self.default_floor_height = use_default_floor_height
        self.hip_ids = [2, 1]
        self.shoulder_ids = [17, 16]
        self.feet_id_r, self.feet_id_l = [8, 11], [7, 10]

    def get_joint_sequence_global_rotation(self, joint_sequence):
        r_hip, l_hip = self.hip_ids
        sdr_r, sdr_l = self.shoulder_ids
        across1 = joint_sequence[:, r_hip] - joint_sequence[:, l_hip]
        across2 = joint_sequence[:, sdr_r] - joint_sequence[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[:, np.newaxis]

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)
        return root_quat

    def align_every_joints(self, global_smplx_joints):
        # align every joint to the global coordinate system
        # smplx_joints shape: (seq_len, 127, 3)
        smplx_joints = deepcopy(global_smplx_joints)
        seq_len, joint_num, _ = smplx_joints.shape
        assert _ == 3
        root_quat = self.get_joint_sequence_global_rotation(smplx_joints)

        smplx_root_trans = deepcopy(smplx_joints[:, 0:1])
        if self.default_floor_height is True:
            smplx_root_trans[:, :, 1] *= 0  # do not move the root in y axis
        else:
            floor_height = smplx_joints.min(axis=0).min(axis=0)[1]
            smplx_root_trans[:, :, 1] = floor_height
        smplx_root_trans *= -1
        # smplx_joints[..., 0] -= smplx_joints[:, 0:1, 0]
        # smplx_joints[..., 2] -= smplx_joints[:, 0:1, 2]
        smplx_joints += smplx_root_trans
        '''All pose face Z+'''
        smplx_joints = qrot_np(np.repeat(root_quat[:, None], smplx_joints.shape[1], axis=1), smplx_joints)
        return smplx_joints, smplx_root_trans, root_quat


    def __call__(self, results: dict) -> dict:
        global_smplx_joints = results['global_smplx_joints']
        if torch.is_tensor(global_smplx_joints):
            global_smplx_joints = global_smplx_joints.cpu().numpy()
        aligned_smplx_joints, smplx_root_trans, root_quat = self.align_every_joints(global_smplx_joints)
        results['aligned_smplx_joints'] = torch.from_numpy(aligned_smplx_joints).float()
        results['smplx_root_trans'] = torch.from_numpy(smplx_root_trans).float()
        results['root_quat'] = torch.from_numpy(root_quat).float()
        return results

@PIPELINES.register_module()
class SplitGlobalSMPLXJoints:
    def __init__(self, smplx_joint_name='aligned_smplx_joints'):
        self.left_hand_dst_idxs, self.left_hand_model_idxs = dset_to_body_model(dset='mano_left', model_type='smplx')
        self.right_hand_dst_idxs, self.right_hand_model_idxs = dset_to_body_model(dset='mano_right',
                                                                                  model_type='smplx')
        self.mo2cap2_dst_idxs, self.smplx_model_idxs = dset_to_body_model(dset='mo2cap2', model_type='smplx')
        self.smplx_joint_name = smplx_joint_name
    def __call__(self, results: dict) -> dict:
        smplx_joints = results[self.smplx_joint_name]
        seq_len, _, __ = smplx_joints.shape
        assert __ == 3

        if isinstance(smplx_joints, np.ndarray):
            smplx_joints = torch.from_numpy(smplx_joints).float()
            results[self.smplx_joint_name] = smplx_joints

        mo2cap2_keypoints_3d = torch.zeros([seq_len, 15, 3]).float()
        mo2cap2_keypoints_3d[:, self.mo2cap2_dst_idxs] = smplx_joints[:, self.smplx_model_idxs]
        left_hand_keypoints_3d = torch.zeros([seq_len, 21, 3]).float()
        left_hand_keypoints_3d[:, self.left_hand_dst_idxs] = smplx_joints[:, self.left_hand_model_idxs]
        right_hand_keypoints_3d = torch.zeros([seq_len, 21, 3]).float()
        right_hand_keypoints_3d[:, self.right_hand_dst_idxs] = smplx_joints[:, self.right_hand_model_idxs]

        results['mo2cap2_keypoints_3d'] = mo2cap2_keypoints_3d
        results['left_hand_keypoints_3d'] = left_hand_keypoints_3d
        results['right_hand_keypoints_3d'] = right_hand_keypoints_3d
        return results

@PIPELINES.register_module()
class ExtractUncertainty:
    def __init__(self, uncertainty_name='uncertainty',
                 target_range=(0.995, 1)):
        self.uncertainty_name = uncertainty_name
        self.target_range = target_range

    def __call__(self, results: dict) -> dict:
        uncertainty = results[self.uncertainty_name]
        if self.target_range is not None:
            # normalize the uncertainty value
            min_uncertainty = uncertainty.min()
            max_uncertainty = uncertainty.max()
            uncertainty = (uncertainty - min_uncertainty) / (max_uncertainty - min_uncertainty)
            # transform it to the target range
            min_target, max_target = self.target_range
            uncertainty = min_target + uncertainty * (max_target - min_target)
        results[self.uncertainty_name] = uncertainty
        return results