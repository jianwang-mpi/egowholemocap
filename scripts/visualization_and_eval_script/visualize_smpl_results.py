import pickle

import numpy as np
import torch
import smplx
from smplx.body_models import build_layer
from mmpose.models.expose.config import cfg
import open3d
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
from mmpose.utils.visualization.skeleton import Skeleton
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated
from mmpose.utils.visualization.draw import draw_joints
import cv2

def main():
    expose_result_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\iknet_only_pose_smpl\results.pkl'
    body_model_path = r'Z:\EgoMocap\work\EgocentricFullBody\models\smplx\SMPLX_NEUTRAL.npz'
    camera_path = r'Z:\EgoMocap\work\EgocentricFullBody\mmpose\utils\fisheye_camera\fisheye.calibration_01_12.json'

    fisheye_camera = FishEyeCameraCalibrated(camera_path)

    body_model_cfg = cfg.get('body_model', {})
    smplx_model = build_layer(
            body_model_path,
            model_type='smplx',
        dtype=torch.float32,
            use_pca=False, flat_hand_mean=True, num_betas=10).cuda()
    smplx_faces = smplx_model.faces

    dset_keyps_idxs, model_keyps_idxs = dset_to_body_model(model_type='smplx', dset='mo2cap2')

    with open(expose_result_path, 'rb') as f:
        smplx_result = pickle.load(f)

    for pred in smplx_result:
        output_body_dict = pred
        output_body_vertices = output_body_dict['vertices']
        output_body_joints = output_body_dict['keypoints_pred']
        img_meta_list = pred['img_metas']
        image_file_list = []
        for img_meta_item in img_meta_list:
            image_file = img_meta_item['image_file']
            image_file_list.append(image_file)
        for i in range(0, len(output_body_vertices), 100):
            print(image_file_list[i])
            vertices = output_body_vertices[i].detach().cpu().numpy()
            # fitted_joints = fitted_smplx_joints_batch[i][:22].cpu().numpy()
            coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
            body_mesh = open3d.geometry.TriangleMesh()
            body_mesh.vertices = open3d.utility.Vector3dVector(vertices)
            body_mesh.triangles = open3d.utility.Vector3iVector(smplx_faces)
            body_mesh.compute_vertex_normals()
            # open3d.visualization.draw(body_mesh, show_skybox=False, show_ui=False)
            open3d.visualization.draw_geometries([body_mesh, coord])

            joints = output_body_joints[i].detach().cpu().numpy()
            joints_mo2cap2 = np.zeros((15, 3))
            joints_mo2cap2[dset_keyps_idxs, :] = joints[model_keyps_idxs, :]

            skeleton = Skeleton(None)
            skeleton_mesh = skeleton.joints_2_mesh(joints_mo2cap2)

            open3d.visualization.draw_geometries([skeleton_mesh, coord])

            # reprojection on 2d images

            joints_2d = fisheye_camera.world2camera(joints)
            image_path = image_file_list[i]
            # conver to local path
            image_path = image_path.replace('/HPS', 'X:')
            img = cv2.imread(image_path)
            img = draw_joints(joints_2d, img)
            cv2.imshow('img', img)
            cv2.waitKey(0)



if __name__ == '__main__':
    main()