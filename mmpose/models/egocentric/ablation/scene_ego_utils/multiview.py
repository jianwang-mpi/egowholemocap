import numpy as np
import torch
# from utils.fisheye.FishEyeCalibrated import FishEyeCameraCalibrated
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated


class Camera:
    def __init__(self, R, t, K, dist=None, name=""):
        self.R = np.array(R).copy()
        assert self.R.shape == (3, 3)

        self.t = np.array(t).copy()
        assert self.t.size == 3
        self.t = self.t.reshape(3, 1)

        self.K = np.array(K).copy()
        assert self.K.shape == (3, 3)

        self.dist = dist
        if self.dist is not None:
            self.dist = np.array(self.dist).copy().flatten()

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])


def euclidean_to_homogeneous(points):
    """Converts euclidean points to homogeneous

    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def project_3d_points_to_image_plane_without_distortion(proj_matrix, points_3d, convert_back_to_euclidean=True):
    """Project 3D points to image plane not taking into account distortion
    Args:
        proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
        points_3d numpy array or torch tensor of shape (N, 3): 3D points
        convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
                                        NOTE: division by zero can be here if z = 0
    Returns:
        numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
    """
    if isinstance(proj_matrix, np.ndarray) and isinstance(points_3d, np.ndarray):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.T
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def project_3d_points_to_image_fisheye_camera(fisheye_camera_model: FishEyeCameraCalibrated,
                                              points_3d):
    """Project 3D points to image plane
        Args:
            fisheye camera model: model of fisheye camera
            points_3d numpy array or torch tensor of shape (N, 3): 3D points
            convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
                                            NOTE: division by zero can be here if z = 0
        Returns:
            numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
    """
    if isinstance(points_3d, np.ndarray):
        result = fisheye_camera_model.world2camera(points_3d)
        return result
    elif torch.is_tensor(points_3d):
        result = fisheye_camera_model.world2camera_pytorch(points_3d)
        return result
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")



def calc_reprojection_error_matrix(keypoints_3d, keypoints_2d_list, proj_matricies):
    reprojection_error_matrix = []
    for keypoints_2d, proj_matrix in zip(keypoints_2d_list, proj_matricies):
        keypoints_2d_projected = project_3d_points_to_image_plane_without_distortion(proj_matrix, keypoints_3d)
        reprojection_error = 1 / 2 * np.sqrt(np.sum((keypoints_2d - keypoints_2d_projected) ** 2, axis=1))
        reprojection_error_matrix.append(reprojection_error)

    return np.vstack(reprojection_error_matrix).T
