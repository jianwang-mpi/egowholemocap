#  Copyright Jian Wang @ MPI-INF (c) 2023.
import os

import open3d

import numpy as np
import json

from natsort import natsorted
from mmpose.utils.visualization.draw import draw_keypoints_3d, draw_skeleton_with_chain

def visualize(xr_egopose_json_path):
    with open(xr_egopose_json_path) as f:
        xr_egopose = json.load(f)
    print(xr_egopose.keys())
    pts3d_fisheye = xr_egopose['pts3d_fisheye']
    pts3d_fisheye = np.asarray(pts3d_fisheye).T
    pts3d_fisheye /= 100

    mesh_pts3d = draw_keypoints_3d(pts3d_fisheye)
    open3d.visualization.draw_geometries([mesh_pts3d])


def main():
    xr_egopose_dir = r'X:\EgoSyn\static00\xr-egopose\male_008_a_a\env_002\cam_down\json'
    json_names = natsorted(os.listdir(xr_egopose_dir))
    for i in range(0, len(json_names), 100):
        json_name = json_names[i]
        json_path = os.path.join(xr_egopose_dir, json_name)
        visualize(json_path)


if __name__ == '__main__':
    main()
