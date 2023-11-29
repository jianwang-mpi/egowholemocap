import numpy as np
import cv2
import torch

from mmpose.models.egocentric.ablation.scene_ego_utils import multiview
from mmpose.utils.visualization.pose_visualization_utils import get_cylinder, get_sphere
import open3d


class Point3D:
    def __init__(self, point, size=3, color=(0, 0, 255)):
        self.point = point
        self.size = size
        self.color = color

    def render(self, proj_matrix, canvas):
        point_2d = multiview.project_3d_points_to_image_plane_without_distortion(
            proj_matrix, np.array([self.point])
        )[0]

        point_2d = tuple(map(int, point_2d))
        cv2.circle(canvas, point_2d, self.size, self.color, self.size)

        return canvas

    def render_open3d(self):
        point_mesh = get_sphere(self.point, radius=0.02, color=self.color)
        return point_mesh



class Line3D:
    def __init__(self, start_point, end_point, size=2, color=(0, 0, 255)):
        self.start_point, self.end_point = start_point, end_point
        self.size = size
        self.color = color

    def render(self, proj_matrix, canvas):
        start_point_2d, end_point_2d = multiview.project_3d_points_to_image_plane_without_distortion(
            proj_matrix, np.array([self.start_point, self.end_point])
        )

        start_point_2d = tuple(map(int, start_point_2d))
        end_point_2d = tuple(map(int, end_point_2d))

        cv2.line(canvas, start_point_2d, end_point_2d, self.color, self.size)

        return canvas

    def render_open3d(self):
        line_mesh = get_cylinder(self.start_point, self.end_point, radius=0.005)
        return line_mesh


class Cuboid3D:
    def __init__(self, position, sides):
        self.position = position
        self.sides = sides

    def build(self):
        primitives = []

        line_color = (255, 255, 0)

        start = self.position + np.array([0, 0, 0])
        primitives.append(Line3D(start, start + np.array([self.sides[0], 0, 0]), color=(255, 0, 0)))
        primitives.append(Line3D(start, start + np.array([0, self.sides[1], 0]), color=(0, 255, 0)))
        primitives.append(Line3D(start, start + np.array([0, 0, self.sides[2]]), color=(0, 0, 255)))

        start = self.position + np.array([self.sides[0], 0, self.sides[2]])
        primitives.append(Line3D(start, start + np.array([-self.sides[0], 0, 0]), color=line_color))
        primitives.append(Line3D(start, start + np.array([0, self.sides[1], 0]), color=line_color))
        primitives.append(Line3D(start, start + np.array([0, 0, -self.sides[2]]), color=line_color))

        start = self.position + np.array([self.sides[0], self.sides[1], 0])
        primitives.append(Line3D(start, start + np.array([-self.sides[0], 0, 0]), color=line_color))
        primitives.append(Line3D(start, start + np.array([0, -self.sides[1], 0]), color=line_color))
        primitives.append(Line3D(start, start + np.array([0, 0, self.sides[2]]), color=line_color))

        start = self.position + np.array([0, self.sides[1], self.sides[2]])
        primitives.append(Line3D(start, start + np.array([self.sides[0], 0, 0]), color=line_color))
        primitives.append(Line3D(start, start + np.array([0, -self.sides[1], 0]), color=line_color))
        primitives.append(Line3D(start, start + np.array([0, 0, -self.sides[2]]), color=line_color))

        return primitives

    def render(self, proj_matrix, canvas):
        # TODO: support rotation

        primitives = self.build()

        for primitive in primitives:
            canvas = primitive.render(proj_matrix, canvas)

        return canvas

    def render_open3d(self):
        primitives = self.build()

        mesh_canvas = []

        for primitive in primitives:
            mesh = primitive.render_open3d()
            mesh_canvas.append(mesh)

        return mesh_canvas


def get_rotation_matrix(axis, theta):
    """Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_coord_volume(coord_volume, theta, axis):
    shape = coord_volume.shape
    device = coord_volume.device

    rot = get_rotation_matrix(axis, theta)
    rot = torch.from_numpy(rot).type(torch.float).to(device)

    coord_volume = coord_volume.view(-1, 3)
    coord_volume = rot.mm(coord_volume.t()).t()

    coord_volume = coord_volume.view(*shape)

    return coord_volume

if __name__ == '__main__':
    cuboid3D = Cuboid3D(position=(-1, -1, 0), sides=(2, 2, 2))

    mesh_list = cuboid3D.render_open3d()
    mesh_list.append(open3d.geometry.TriangleMesh.create_coordinate_frame())

    open3d.visualization.draw_geometries(mesh_list)
