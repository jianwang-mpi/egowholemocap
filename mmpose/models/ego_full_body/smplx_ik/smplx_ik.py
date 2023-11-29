#  Copyright Jian Wang @ MPI-INF (c) 2023.
import json
import os.path
import pickle
import random
from os import path as osp

import numpy as np
import open3d
import torch
from colour import Color
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.ik_engine import IK_Engine
from torch import nn

from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model


# from visualization_utils import visualize_joint_and_mesh_single


def joint_to_spheres(joints, color=(1, 0, 0), radius=0.01):
    point_list = []
    for body_joint in joints:
        joint_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
        joint_sphere.translate(body_joint)
        joint_sphere.paint_uniform_color(color)
        point_list.append(joint_sphere)
    return point_list


def visualize_joint_and_mesh_single(bm_fname, ik_res_single):
    if not isinstance(bm_fname, BodyModel):
        bm_vis = BodyModel(bm_fname, num_betas=16).to(ik_res_single['pose_body'].device)
    else:
        bm_vis = bm_fname

    fitted_human_body_batch = bm_vis(**{
        'pose_body': ik_res_single['pose_body'],
        'root_orient': ik_res_single['root_orient'],
        'trans': ik_res_single['trans'],
        'betas': ik_res_single['betas']
    })
    target_joints_batch = ik_res_single['target_pts']
    fitted_smplx_joints_batch = fitted_human_body_batch.Jtr

    faces = fitted_human_body_batch.f.cpu().numpy()
    vertices_batch = fitted_human_body_batch.v
    assert len(vertices_batch) == len(target_joints_batch)
    for i in range(0, len(vertices_batch), 30):
        vertices = vertices_batch[i].cpu().numpy()
        target_joints = target_joints_batch[i].cpu().numpy()
        fitted_joints = fitted_smplx_joints_batch[i].cpu().numpy()

        body_mesh = open3d.geometry.TriangleMesh()
        body_mesh.vertices = open3d.utility.Vector3dVector(vertices)
        body_mesh.triangles = open3d.utility.Vector3iVector(faces)
        body_mesh.compute_vertex_normals()
        # open3d.visualization.draw(body_mesh, show_skybox=False, show_ui=False)

        body_mesh_lineset = open3d.geometry.LineSet.create_from_triangle_mesh(body_mesh)
        target_points = joint_to_spheres(target_joints)

        fitted_points = joint_to_spheres(fitted_joints, color=(0, 1, 0))

        open3d.visualization.draw_geometries([body_mesh_lineset] + target_points + fitted_points)


def extract_body_pose(bm_fname, ik_res_single):
    if not isinstance(bm_fname, BodyModel):
        bm_vis = BodyModel(bm_fname, num_betas=16).to(ik_res_single['pose_body'].device)
    else:
        bm_vis = bm_fname

    fitted_human_body_batch = bm_vis(**{
        'pose_body': ik_res_single['pose_body'],
        'root_orient': ik_res_single['root_orient'],
        'trans': ik_res_single['trans'],
        'betas': ik_res_single['betas']
    })
    fitted_smplx_joints_batch = fitted_human_body_batch.Jtr
    # ik_res_single['fitted_smplx_joints'] = fitted_smplx_joints_batch
    return fitted_smplx_joints_batch


class Mo2Cap2WithHeadKeyPoints(nn.Module):
    def __init__(self,
                 bm_fname,
                 num_betas=16,
                 n_joints=16,
                 ):
        super(Mo2Cap2WithHeadKeyPoints, self).__init__()
        self.bm = BodyModel(bm_fname, num_betas=num_betas, persistant_buffer=False)
        self.dset_keyps_idxs, self.model_keyps_idxs = dset_to_body_model(model_type='smplx', dset='mo2cap2_with_head')
        self.bm_f = []  # self.bm.f
        self.kpts_colors = np.array([Color('grey').rgb for _ in range(n_joints)])
        self.n_joints = n_joints

    def forward(self, body_parms):
        new_body = self.bm(**body_parms)

        source_kpts_smplx = new_body.Jtr
        source_kpts_mo2cap2 = torch.zeros((source_kpts_smplx.shape[0], self.n_joints, 3)).float().to(
            source_kpts_smplx.device)

        source_kpts_mo2cap2[:, self.dset_keyps_idxs, :] = source_kpts_smplx[:, self.model_keyps_idxs, :]

        return {'source_kpts': source_kpts_mo2cap2, 'body': new_body}


def load_egocentric_start_frame(syn_json):
    with open(syn_json, 'r') as f:
        data = json.load(f)
    return data['ego']


def save_mesh(bm_fname, result_list, save_dir, comp_device):
    if not isinstance(bm_fname, BodyModel):
        bm_vis = BodyModel(bm_fname, num_betas=16).to(comp_device)
    else:
        bm_vis = bm_fname

    for i, result in enumerate(result_list):
        fitted_human_body_batch = bm_vis(**{
            'pose_body': result['pose_body'].unsqueeze(0),
            'root_orient': result['root_orient'].unsqueeze(0),
            'trans': result['trans'].unsqueeze(0),
            'betas': result['betas'].unsqueeze(0)
        })
        faces = fitted_human_body_batch.f.cpu().numpy()
        vertices = fitted_human_body_batch.v.cpu().numpy()[0]
        body_mesh = open3d.geometry.TriangleMesh()
        body_mesh.vertices = open3d.utility.Vector3dVector(vertices)
        body_mesh.triangles = open3d.utility.Vector3iVector(faces)
        body_mesh.compute_vertex_normals()

        save_path = os.path.join(save_dir, 'pose_%06d.ply' % i)

        open3d.io.write_triangle_mesh(save_path, body_mesh)

class SMPLX_IK:
    def __init__(self):
        if os.name == 'nt':
            support_dir = r'Z:\EgoMocap\work\EgocentricFullBody\3rdparty\human_body_prior\support_data\downloads'
        else:
            support_dir = '/CT/EgoMocap/work/EgocentricFullBody/3rdparty/human_body_prior/support_data/downloads'
        vposer_expr_dir = osp.join(support_dir, 'V02_05')
        self.bm_fname = osp.join(support_dir, 'models_lockedhead/smplx/SMPLX_NEUTRAL.npz')

        self.comp_device = torch.device('cuda')

        data_loss = torch.nn.MSELoss(reduction='sum')

        stepwise_weights = [{'data': 10., 'poZ_body': 0.000001, 'betas': .005}]

        optimizer_args = {'type': 'LBFGS', 'max_iter': 500, 'lr': 2, 'tolerance_change': 1e-5, 'history_size': 300}
        self.ik_engine = IK_Engine(vposer_expr_dir=vposer_expr_dir,
                              verbosity=2,
                              display_rc=(2, 2),
                              data_loss=data_loss,
                              stepwise_weights=stepwise_weights,
                              optimizer_args=optimizer_args).to(self.comp_device)
        self.source_pts = Mo2Cap2WithHeadKeyPoints(bm_fname=self.bm_fname).to(self.comp_device)

    def ik(self, pose_batch: torch.Tensor):
        batch_size = len(pose_batch)

        target_pts = pose_batch.float().to(self.comp_device)
        # add head joint to target keypoints
        head_joint_position = torch.asarray([[[0, 0.255, 0.04]]])
        head_joint_position = torch.repeat_interleave(head_joint_position, batch_size, dim=0).to(self.comp_device)
        target_pts = torch.cat([head_joint_position, target_pts], dim=1)


        ik_res = self.ik_engine(self.source_pts, target_pts)
        ik_res_detached = {k: v.detach() for k, v in ik_res.items()}
        ik_res_detached['target_pts'] = target_pts
        # get smplx joints
        ik_res_detached['fitted_smplx_joints'] = extract_body_pose(self.bm_fname, ik_res_detached)

        # for i, frame_id in enumerate(rnd_frame_ids):
        #     result_list[frame_id] = {}
        #     for k, v in ik_res_detached.items():
        #         result_list[frame_id][k] = v[i]
        #         result_list[frame_id]['target_pts'] = target_pts[i]
        #
        # if visualize:
        #     print('visualize')
        #     bm_vis = BodyModel(self.bm_fname, num_betas=16).to(self.comp_device)
        #     for result_item in result_list:
        #         if result_item is not None:
        #             visualize_joint_and_mesh_single(bm_vis, result_item)

        return ik_res_detached




if __name__ == '__main__':
    # main(visualize=False, save_path='out')
    if os.name == 'nt':
        full_body_save_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_b_256\results.pkl'
        save_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\egofullbody_test_b_256\results_smplx_ik.pkl'
        visualize = True
    else:
        full_body_save_path = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/egofullbody_test_b_256/results.pkl'
        save_path = '/CT/EgoMocap/work/EgocentricFullBody/work_dirs/egofullbody_test_b_256/results_smplx_ik.pkl'
        visualize = False

    with open(full_body_save_path, 'rb') as f:
        full_body_results = pickle.load(f)
    smplx_ik = SMPLX_IK()
    for i, pred in enumerate(full_body_results):
        pred_joints_3d = pred['body_pose_results']['keypoints_pred']

        if not torch.is_tensor(pred_joints_3d):
            pred_joints_3d = torch.from_numpy(pred_joints_3d).cuda()
        ik_res = smplx_ik.ik(pred_joints_3d)

        full_body_results[i]['smplx_ik'] = ik_res

        if visualize:
            print('visualize')
            bm_vis = BodyModel(smplx_ik.bm_fname, num_betas=16).to(smplx_ik.comp_device)
            visualize_joint_and_mesh_single(bm_vis, ik_res)

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(full_body_results, f)
