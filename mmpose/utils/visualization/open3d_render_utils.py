#  Copyright Jian Wang @ MPI-INF (c) 2023.
import open3d
import os
from natsort import natsorted
from tqdm import tqdm

def render_open3d(mesh_list, viewpoint, out_path):
    vis = open3d.visualization.Visualizer()
    vis.create_window(visible=False)
    ctr = vis.get_view_control()
    param = open3d.io.read_pinhole_camera_parameters(viewpoint)
    for mesh in mesh_list:
        vis.add_geometry(mesh)
    ctr.convert_from_pinhole_camera_parameters(param)
    # vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path)
    vis.destroy_window()

def save_view_point(mesh_list, filename):
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    for mesh in mesh_list:
        vis.add_geometry(mesh)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()