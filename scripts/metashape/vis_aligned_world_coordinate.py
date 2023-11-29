#  Copyright Jian Wang @ MPI-INF (c) 2023.

import json
import numpy as np
from copy import copy

import open3d
import torch

import os
import pickle
import xml.etree.ElementTree as ET

from natsort import natsorted
from tqdm import tqdm


def calcualate_depth_scale(depth_scale_json_file, log_err=False):
    with open(depth_scale_json_file, 'r') as f:
        depth_scale_data_list = json.load(f)
    # print(depth_scale_data_list)

    scale_list = []
    for scale_data in depth_scale_data_list:
        x1 = scale_data['x1']
        x2 = scale_data['x2']

        x1 = np.asarray(x1)
        x2 = np.asarray(x2)

        distance = np.linalg.norm(x2 - x1)
        # print(distance)
        scale = scale_data['real'] / distance
        scale_list.append(scale)
    if log_err:
        print(scale_list)
        print(np.std(scale_list) / np.average(scale_list))
    # print(np.average(scale_list))
    return np.average(scale_list)

def camera_to_matrix(camera_text):
    matrix = np.zeros(shape=(4, 4))
    camera_text_list = camera_text.split()
    for i in range(len(camera_text_list)):
        row = i // 4
        column = i % 4
        matrix[row][column] = float(camera_text_list[i])
    return matrix

def transform_to_matrix(rot_text, trans_text):
    matrix = np.zeros(shape=(4, 4))
    matrix[3, 3] = 1
    rot_text_list = rot_text.split()
    for i in range(len(rot_text_list)):
        row = i // 3
        column = i % 3
        matrix[row][column] = float(rot_text_list[i])
    trans_text_split = trans_text.split()
    matrix[0, 3] = float(trans_text_split[0])
    matrix[1, 3] = float(trans_text_split[1])
    matrix[2, 3] = float(trans_text_split[2])
    return matrix

def get_camera_pose(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # get global transform if available
    global_transform_component = root.find('chunk').find('components').find('component')
    transform_info = global_transform_component.find('transform')
    rotation_info = transform_info.find('rotation').text
    translation_info = transform_info.find('translation').text
    # scale_info = transform_info.find('scale').text

    global_transform_matrix = transform_to_matrix(rotation_info, translation_info)

    camera_list = {}
    for camera in tqdm(root.iter('camera')):
        # print(camera.attrib)
        # print(camera[0].text)
        if len(camera) < 1:
            continue
        matrix = camera_to_matrix(camera[0].text)

        matrix = global_transform_matrix.dot(matrix)

        image_name = camera.attrib['label']

        camera_list[image_name] = matrix
    return camera_list

def load_gt_scene(scene_path, scale):
    scene_geometry_mesh = open3d.io.read_triangle_mesh(scene_path)
    scene_geometry_mesh.scale(scale, center=(0, 0, 0))
    return scene_geometry_mesh


def main():
    scene_base_path = r'Z:\EgoMocap\work\EgoBodyInContext\sfm_data\kripa\out'
    scene_path = os.path.join(scene_base_path, 'mesh_floor_aligned', 'scene_floor_aligned.obj')
    scene_scale_path = os.path.join(scene_base_path, 'scale.json')
    scale = calcualate_depth_scale(scene_scale_path)

    print('scale: {}'.format(scale))

    scene_geometry_mesh = load_gt_scene(scene_path, scale)

    camera_xml_path = os.path.join(scene_base_path, 'mesh_floor_aligned', 'camera_floor_aligned_2.xml')

    camera_list = get_camera_pose(camera_xml_path)

    coord = open3d.geometry.TriangleMesh.create_coordinate_frame()

    camera_list_names = natsorted(camera_list.keys())
    for i in range(0, len(camera_list_names), 50):
        camera_name = camera_list_names[i]
        camera_pose = camera_list[camera_name]
        camera_pose[:3, 3] = camera_pose[:3, 3] * scale
        camera_coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
        camera_coord.transform(camera_pose)
        coord += camera_coord

    open3d.visualization.draw_geometries([scene_geometry_mesh, coord])

if __name__ == '__main__':
    main()