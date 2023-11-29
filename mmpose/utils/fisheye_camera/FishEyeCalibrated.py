import json
import numpy as np
import torch
from copy import deepcopy


class FishEyeCameraCalibrated:
    def __init__(self, calibration_file_path, use_gpu=False):
        with open(calibration_file_path) as f:
            calibration_data = json.load(f)
        self.intrinsic = np.array(calibration_data['intrinsic'])
        self.img_size = np.array(calibration_data['size'])  # w, h
        self.fisheye_polynomial = np.array(calibration_data['polynomialC2W'])
        self.fisheye_inverse_polynomial = np.array(calibration_data['polynomialW2C'])
        self.img_center = np.array([self.intrinsic[0][2], self.intrinsic[1][2]])
        self.use_gpu = use_gpu
        # self.img_center = np.array([self.img_size[0] / 2, self.img_size[1] / 2])

    def camera2world(self, point: np.ndarray, depth: np.ndarray):
        """
        point: np.ndarray of 2D points on image (n * 2)
        depth: np.ndarray of depth of every 2D points (n)
        """
        depth = depth.astype(np.float32)
        point_centered = point.astype(np.float32) - self.img_center
        x = point_centered[:, 0]
        y = point_centered[:, 1]
        distance_from_center = np.sqrt(np.square(x) + np.square(y))

        z = np.polyval(p=self.fisheye_polynomial[::-1], x=distance_from_center)
        point_3d = np.array([x, y, -z])  # 3, n
        norm = np.linalg.norm(point_3d, axis=0)
        point_3d = point_3d / norm * depth
        return point_3d.transpose()

    def camera2world_ray(self, point: np.ndarray):
        """
        calculate the ray direction from the image points
        point: np.ndarray of 2D points on image (n * 2)
        depth: np.ndarray of depth of every 2D points (n)
        """
        point_centered = point.astype(np.float) - self.img_center
        x = point_centered[:, 0]
        y = point_centered[:, 1]
        distance_from_center = np.sqrt(np.square(x) + np.square(y))

        z = np.polyval(p=self.fisheye_polynomial[::-1], x=distance_from_center)
        point_3d = np.array([x, y, -z])  # 3, n
        norm = np.linalg.norm(point_3d, axis=0)
        point_3d = point_3d / norm
        return point_3d.transpose()

    def getPolyVal(self, p, x):
        curVal = torch.zeros_like(x)
        for curValIndex in range(len(p) - 1):
            curVal = (curVal + p[curValIndex]) * x
        return curVal + p[len(p) - 1]

    def camera2world_pytorch(self, point: torch.Tensor, depth: torch.Tensor):
        """
                point: np.ndarray of 2D points on image (n * 2)
                depth: np.ndarray of depth of every 2D points (n)
                """
        img_center = torch.from_numpy(self.img_center).float().to(point.device)
        point_centered = point.float() - img_center
        x = point_centered[:, 0]
        y = point_centered[:, 1]
        distance_from_center = torch.sqrt(torch.square(x) + torch.square(y))

        z = self.getPolyVal(p=self.fisheye_polynomial[::-1], x=distance_from_center)
        point_3d = torch.stack([x, y, -z]).float()  # 3, n
        norm = torch.norm(point_3d, dim=0)
        point_3d = point_3d / norm * depth.float()
        return point_3d.t()

    def camera2world_ray_pytorch(self, point: torch.Tensor):
        """
                point: np.ndarray of 2D points on image (n * 2)
                depth: np.ndarray of depth of every 2D points (n)
                """
        point_centered = point.float() - self.img_center
        x = point_centered[:, 0]
        y = point_centered[:, 1]
        distance_from_center = torch.sqrt(torch.square(x) + torch.square(y))

        z = self.getPolyVal(p=self.fisheye_polynomial[::-1], x=distance_from_center)
        point_3d = torch.stack([x, y, -z]).float()  # 3, n
        norm = torch.norm(point_3d, dim=0)
        point_3d = point_3d / norm
        return point_3d.t()

    def world2camera(self, point3D):
        point3D = deepcopy(point3D)
        point3D[:, 2] = point3D[:, 2] * -1
        point3D = point3D.T
        xc, yc = self.img_center[0], self.img_center[1]
        point2D = []

        norm = np.linalg.norm(point3D[:2], axis=0)

        if (norm != 0).all():
            theta = np.arctan(point3D[2] / norm)
            invnorm = 1.0 / norm
            t = theta
            rho = self.fisheye_inverse_polynomial[0]
            t_i = 1.0

            for i in range(1, len(self.fisheye_inverse_polynomial)):
                t_i *= t
                rho += t_i * self.fisheye_inverse_polynomial[i]

            x = point3D[0] * invnorm * rho
            y = point3D[1] * invnorm * rho

            point2D.append(x + xc)
            point2D.append(y + yc)
        else:
            point2D.append(xc)
            point2D.append(yc)
            raise Exception("norm is zero!")

        return np.asarray(point2D).T

    def world2camera_with_depth(self, point3D):
        point3D_cloned = deepcopy(point3D)
        point2D = self.world2camera(point3D_cloned)

        depth = np.linalg.norm(point3D, axis=-1)
        return point2D, depth

    def world2camera_pytorch_with_depth(self, point3D):
        point2D = self.world2camera_pytorch(point3D)

        depth = torch.norm(point3D, dim=-1)
        return point2D, depth

    def world2camera_pytorch(self, point3d_original: torch.Tensor, normalize=False):
        """

        Args:
            point3d_original: point
            normalize: normalize to -1, 1

        Returns:

        """
        fisheye_inv_polynomial = self.fisheye_inverse_polynomial
        point3d = point3d_original.clone()
        point3d[:, 2] = point3d_original[:, 2] * -1
        point3d = point3d.transpose(0, 1)
        xc, yc = self.img_center[0], self.img_center[1]
        xc = torch.Tensor([xc]).float().to(point3d.device)
        yc = torch.Tensor([yc]).float().to(point3d.device)
        point2d = torch.empty((2, point3d.shape[-1])).to(point3d.device)

        norm = torch.norm(point3d[:2], dim=0)

        if (norm != 0).all():
            theta = torch.atan(point3d[2] / norm)
            invnorm = 1.0 / norm
            t = theta
            rho = fisheye_inv_polynomial[0]
            t_i = 1.0

            for i in range(1, len(fisheye_inv_polynomial)):
                t_i *= t
                rho += t_i * fisheye_inv_polynomial[i]

            x = point3d[0] * invnorm * rho
            y = point3d[1] * invnorm * rho

            point2d[0] = x + xc
            point2d[1] = y + yc
        else:
            point2d[0] = xc
            point2d[1] = yc
            raise Exception("norm is zero!")

        # if normalize, the point result will be -1 to 1from
        if normalize is True:
            image_w, image_h = self.img_size[0], self.img_size[1]
            assert image_w > image_h
            point2d[0] = point2d[0] - (image_w - image_h) // 2  # to square image
            point2d = point2d / (image_h - 1) * 2   # to [0, 2]
            point2d -= 1

        return point2d.transpose(0, 1)

    def undistort(self, point_2Ds):
        """
        undistort the input 2d points in fisheye camera
        """
        point_length = point_2Ds.shape[0]
        depths = np.ones(shape=point_length)
        point_3Ds = self.camera2world(point_2Ds, depths)

        # point_3Ds_homo = np.ones((point_length, 4))
        # point_3Ds_homo[:, :3] = point_3Ds

        projected_2d_points = (self.intrinsic[:3, :3] @ point_3Ds.T).T
        projected_2d_points = projected_2d_points[:, :2] / projected_2d_points[:, 2:]
        return projected_2d_points


if __name__ == '__main__':
    camera = FishEyeCameraCalibrated(r'Z:\EgoMocap\work\EgocentricFullBody\mmpose\utils\fisheye_camera\fisheye.calibration_01_12.json')
    point = np.array([[660, 520], [520, 660], [123, 456]])
    depth = np.array([30, 30, 40])
    point = torch.from_numpy(point).cuda()
    depth = torch.from_numpy(depth).cuda()
    point3d = camera.camera2world_pytorch(point, depth)
    print(point3d)

    # reprojected_point_2d = camera.world2camera(point3d)
    # print(reprojected_point_2d)

    reprojected_point_2d = camera.world2camera_pytorch(point3d.cpu(), normalize=False)
    print(reprojected_point_2d)
