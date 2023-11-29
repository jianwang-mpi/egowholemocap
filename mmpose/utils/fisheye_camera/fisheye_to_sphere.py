# project fisheye image to a sphere mesh
import math

import cv2
import numpy as np
import open3d
from scipy.spatial.transform import Rotation as R
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated


class Fisheye2Sphere:
    def __init__(self, fisheye_camera_param_path, img_h=1024, img_w=1280):
        # build fisheye image coordinates

        self.fisheye_camera = FishEyeCameraCalibrated(fisheye_camera_param_path)

        # build 3d coords
        x = np.arange(0, img_w, 1)
        y = np.arange(0, img_h, 1)
        self.coord_2d_raw = []
        for x_i in x:
            for y_i in y:
                self.coord_2d_raw.append([x_i, y_i])
        self.coord_2d_raw = np.asarray(self.coord_2d_raw)
        self.coord_3d_raw = self.fisheye_camera.camera2world(self.coord_2d_raw, np.ones(self.coord_2d_raw.shape[0]))

        correct_indices = self.coord_3d_raw[:, 2] > -0.1
        self.coord_2d = self.coord_2d_raw[correct_indices]
        self.coord_2d = self.coord_2d.astype(np.int32)
        self.coord_3d = self.coord_3d_raw[correct_indices]

    def calculate_crop_circle_radius(self):
        coord_2d_centered = self.coord_2d.astype(np.float32) - self.fisheye_camera.img_center
        radius = np.linalg.norm(coord_2d_centered, axis=1)
        print(np.max(radius))

    @staticmethod
    def visualize_point_cloud(coord_3d):
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(coord_3d)

        open3d.visualization.draw_geometries([point_cloud])

    def fisheye_to_sphere(self, fisheye_image):
        fisheye_image = cv2.imread(fisheye_image)
        fisheye_image = fisheye_image[:, :, ::-1]

        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(self.coord_3d)

        # colors = fisheye_image[self.coord_2d[:, ::-1]]
        colors = []
        for coord in self.coord_2d:
            colors.append(fisheye_image[coord[1], coord[0]] / 255.)
        point_cloud.colors = open3d.utility.Vector3dVector(colors)

        # open3d.visualization.draw_geometries([point_cloud])
        open3d.io.write_point_cloud(r'Z:\EgoMocap\work\EgocentricFullBody\vis_results_paper\sphere_paper.ply', point_cloud)

    def polar2cart(self, r, lat, lon):
        return [
            r * math.sin(lat) * math.cos(lon),
            r * math.sin(lat) * math.sin(lon),
            r * math.cos(lat)
        ]

    def generate_patch_coordinates(self, patch_center_lat, patch_center_lon, patch_size, patch_pixel_num, visualize=False):
        # generate (x, y) pairs
        x_range = np.arange(0, patch_pixel_num[0]) / (patch_pixel_num[0] - 1) - 0.5
        y_range = np.arange(0, patch_pixel_num[1]) / (patch_pixel_num[1] - 1) - 0.5

        x_range *= 2
        y_range *= 2

        x_range *= patch_size[0]
        y_range *= patch_size[1]

        patch_centers = np.dstack(np.meshgrid(x_range, y_range)).reshape(-1, 2)
        patch_centers = np.concatenate([patch_centers, np.ones((len(patch_centers), 1))], axis=1)
        r_lat = R.from_euler('xyz', (patch_center_lat, 0, 0), degrees=False)
        r_lon = R.from_euler('xyz', (0, 0, patch_center_lon), degrees=False)
        rotated_patch_centers = r_lat.apply(patch_centers)
        rotated_patch_centers = r_lon.apply(rotated_patch_centers)

        if visualize:
            sphere_list = []
            coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
            for patch_center in rotated_patch_centers:
                sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(patch_center)
                sphere_list.append(sphere)
            open3d.visualization.draw_geometries(sphere_list + [coord])

        rotated_patch_centers = rotated_patch_centers.reshape((len(x_range), len(y_range), 3))
        return rotated_patch_centers
        #
        # coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
        # for sphere in sphere_list:
        #     sphere.rotate(open3d.geometry.get_rotation_matrix_from_xyz((patch_center_lat, 0, 0)), center=(0, 0, 0))
        #     sphere.rotate(open3d.geometry.get_rotation_matrix_from_xyz((0, 0, patch_center_lon)), center=(0, 0, 0))
        #     coord += sphere
        # # open3d.visualization.draw_geometries([coord])
        # open3d.io.write_triangle_mesh(r'Z:\EgoMocap\work\EgocentricFullBody\vis_results\patch_moving\patch_%03d.ply' % patch_id, coord)

    def generate_sphere_patch(self, patch_number=(20, 10),
                              patch_size=(0.3, 0.3), patch_pixel_number=(64, 64)):
        """generate patches on sphere
        Given a center point and patch size, get the points of the patch and project it to 2d image

        params:
        """
        width, height = patch_number
        h_range = np.arange(1, height, 1)
        w_range = np.arange(0, width, 1)
        lat_range = (h_range / height) * np.pi / 2  # 0 ~ pi/2
        lon_range = ((w_range / width) - 0.5) * 2 * np.pi  # -pi ~ pi
        print(lat_range)
        print(lon_range)

        patch_centers = np.dstack(np.meshgrid(lat_range, lon_range)).reshape(-1, 2)

        patch_center_cart_list = []
        patch_coordinate_list = []
        for i, patch_center in enumerate(patch_centers):
            coordinate_path_center = self.polar2cart(1, patch_center[0], patch_center[1])
            patch_center_cart_list.append(coordinate_path_center)

            patch_coordinates = self.generate_patch_coordinates(patch_center[0], patch_center[1],
                                                                patch_size, patch_pixel_number)
            patch_coordinate_list.append(patch_coordinates)

        # sphere_list = []
        # for patch_center_cart in patch_center_cart_list:
        #     sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        #     sphere.translate(patch_center_cart)
        #     sphere_list.append(sphere)
        # open3d.visualization.draw_geometries(sphere_list)
        return patch_coordinate_list

    def get_pixel_from_image(self, image, patch_coordinate_list):
        for image_i, patch in enumerate(patch_coordinate_list):
            image_patch = np.zeros_like(patch)
            H, W, _ = patch.shape
            patch_flat = patch.reshape((-1, 3))
            pos_2d_flat = self.fisheye_camera.world2camera(patch_flat)
            pose_2d_list = pos_2d_flat.reshape((H, W, 2))
            for i in range(H):
                for j in range(W):
                    image_patch[i][j] = image[int(pose_2d_list[i][j][1]), int(pose_2d_list[i][j][0]), :]

            image_patch = image_patch.astype(np.uint8)
            out_path = r'Z:\EgoMocap\work\EgocentricFullBody\vis_results\patch_moving_image\img_%03d.jpg' % image_i
            cv2.imwrite(out_path, image_patch)
            # cv2.imshow('patch', image_patch)
            # cv2.waitKey(0)


if __name__ == '__main__':
    fisheye2sphere = Fisheye2Sphere(
        r'Z:\EgoMocap\work\EgocentricFullBody\mmpose\utils\fisheye_camera\fisheye.calibration_01_12.json')
    # fisheye2sphere.fisheye_to_sphere(r'X:\ScanNet\work\egocentric_view\25082022\jian1\imgs\img_000000.jpg')
    # fisheye2sphere.calculate_crop_circle_radius()
    patches = fisheye2sphere.generate_sphere_patch()
    print('patch generated')
    image = cv2.imread(r'X:\ScanNet\work\egocentric_view\25082022\jian1\imgs\img_000000.jpg')
    fisheye2sphere.get_pixel_from_image(image, patches)

