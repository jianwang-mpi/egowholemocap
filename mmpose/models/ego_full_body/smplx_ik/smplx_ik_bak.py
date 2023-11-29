#  Copyright Jian Wang @ MPI-INF (c) 2023.
import json
import os.path
import pickle
import open3d
import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel
from torch import nn
from colour import Color
from human_body_prior.models.ik_engine import IK_Engine
from os import path as osp
# from visualization_utils import visualize_joint_and_mesh_single
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model


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
        'pose_body': ik_res_single['pose_body'].unsqueeze(0),
        'root_orient': ik_res_single['root_orient'].unsqueeze(0),
        'trans': ik_res_single['trans'].unsqueeze(0),
        'betas': ik_res_single['betas'].unsqueeze(0)
    })
    target_joints_batch = ik_res_single['target_pts'].unsqueeze(0)
    fitted_smplx_joints_batch = fitted_human_body_batch.Jtr

    faces = fitted_human_body_batch.f.cpu().numpy()
    vertices_batch = fitted_human_body_batch.v
    assert len(vertices_batch) == len(target_joints_batch)
    for i in range(len(vertices_batch)):
        vertices = vertices_batch[i].cpu().numpy()
        target_joints = target_joints_batch[i].cpu().numpy()
        fitted_joints = fitted_smplx_joints_batch[i][:22].cpu().numpy()

        body_mesh = open3d.geometry.TriangleMesh()
        body_mesh.vertices = open3d.utility.Vector3dVector(vertices)
        body_mesh.triangles = open3d.utility.Vector3iVector(faces)
        body_mesh.compute_vertex_normals()
        # open3d.visualization.draw(body_mesh, show_skybox=False, show_ui=False)

        body_mesh_lineset = open3d.geometry.LineSet.create_from_triangle_mesh(body_mesh)
        target_points = joint_to_spheres(target_joints)

        fitted_points = joint_to_spheres(fitted_joints, color=(0, 1, 0))

        open3d.visualization.draw_geometries([body_mesh_lineset] + target_points + fitted_points)

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
        source_kpts_mo2cap2 = torch.zeros((source_kpts_smplx.shape[0], self.n_joints, 3)).float().to(source_kpts_smplx.device)

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


def main(visualize=False, save_path=None):
    if os.name == 'nt':
        support_dir = r'Z:\EgoMocap\work\EgocentricFullBody\3rdparty\human_body_prior\support_data\downloads'
    else:
        support_dir = '/CT/EgoMocap/work/EgocentricFullBody/3rdparty/human_body_prior/support_data/downloads'
    vposer_expr_dir = osp.join(support_dir, 'V02_05')
    bm_fname = osp.join(support_dir,'models_lockedhead/smplx/SMPLX_NEUTRAL.npz')

    comp_device = torch.device('cuda')


    motion_data_path = r'X:\ScanNet\work\egocentric_view\25082022\jian2\out\egopw_results.pkl'
    egocentric_start_frame = load_egocentric_start_frame(os.path.join(os.path.split(motion_data_path)[0], '../syn.json'))
    with open(motion_data_path, 'rb') as f:
        pred_pose_list = pickle.load(f)
        if 'egopw' in motion_data_path:
            egopw_joint_dict = pred_pose_list['estimated_local_skeleton']
            pred_pose_list = [egopw_joint_dict['img_%06d.jpg' % i] for i in range(len(egopw_joint_dict))]
            pred_pose_list = pred_pose_list[egocentric_start_frame:]
        pred_pose_list = np.asarray(pred_pose_list)

    # start_frame = 1000
    # pred_pose_list = pred_pose_list[start_frame:32 + start_frame]

    # create source and target key points and make sure they are index aligned
    data_loss = torch.nn.MSELoss(reduction='sum')

    stepwise_weights = [{'data': 10., 'poZ_body': 0.000001, 'betas': .005}]

    optimizer_args = {'type':'LBFGS', 'max_iter':500, 'lr':2, 'tolerance_change': 1e-5, 'history_size':300}
    ik_engine = IK_Engine(vposer_expr_dir=vposer_expr_dir,
                          verbosity=2,
                          display_rc= (2, 2),
                          data_loss=data_loss,
                          stepwise_weights=stepwise_weights,
                          optimizer_args=optimizer_args).to(comp_device)

    from human_body_prior.tools.omni_tools import create_list_chunks
    frame_ids = np.arange(len(pred_pose_list))
    np.random.shuffle(frame_ids)
    batch_size = 256

    result_list = [None] * len(pred_pose_list)
    source_pts = Mo2Cap2WithHeadKeyPoints(bm_fname=bm_fname).to(comp_device)

    for rnd_frame_ids in create_list_chunks(frame_ids, batch_size, overlap_size=0, cut_smaller_batches=False):
        print(rnd_frame_ids)
        target_pts = torch.from_numpy(pred_pose_list[rnd_frame_ids, :, :]).float().to(comp_device)
        # add head joint to target keypoints
        head_joint_position = torch.asarray([[[0, 0.255, 0.04]]])
        head_joint_position = torch.repeat_interleave(head_joint_position, len(rnd_frame_ids), dim=0).to(comp_device)
        target_pts = torch.cat([head_joint_position, target_pts], dim=1)


        ik_res = ik_engine(source_pts, target_pts)
        ik_res_detached = {k: v.detach() for k, v in ik_res.items()}

        # ik_res = ik_engine(source_pts, target_pts, initial_body_params=ik_res_detached,
        #                    free_var_names=('betas', 'trans', 'root_orient'))
        # ik_res_detached = {k: v.detach() for k, v in ik_res.items()}

        for i, frame_id in enumerate(rnd_frame_ids):
            result_list[frame_id] = {}
            for k, v in ik_res_detached.items():
                result_list[frame_id][k] = v[i]
                result_list[frame_id]['target_pts'] = target_pts[i]


        if visualize:
            print('visualize')
            bm_vis = BodyModel(bm_fname, num_betas=16).to(comp_device)
            for result_item in result_list:
                if result_item is not None:
                    visualize_joint_and_mesh_single(bm_vis, result_item)

    if save_path is not None:
        file_name = os.path.split(motion_data_path)[1]
        with open(os.path.join(save_path, file_name), 'wb') as f:
            pickle.dump(result_list, f)
        # bm_vis = BodyModel(bm_fname, num_betas=16).to(comp_device)
        # save_mesh(bm_vis, result_list, save_path, comp_device)

if __name__ == '__main__':
    # main(visualize=False, save_path='out')
    main(visualize=True, save_path=None)
