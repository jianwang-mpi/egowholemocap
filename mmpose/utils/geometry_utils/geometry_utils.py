#  Copyright Jian Wang @ MPI-INF (c) 2023.
import torch
import numpy as np
import mmpose.utils.geometry_utils.geometry_utils_torch as gut


def __angle_axis_to_rotation_matrix_torch(aa):
    aa = aa.clone()
    if aa.dim() == 1:
        assert aa.size(0) ==3
        aa = aa.view(1, 3)
        rotmat = gut.angle_axis_to_rotation_matrix(aa)[0][:3, :3]
    elif aa.dim() == 2:
        assert aa.size(1) == 3
        rotmat = gut.angle_axis_to_rotation_matrix(aa)[:, :3, :3]
    else:
        assert aa.dim() == 3
        dim0 = aa.size(0)
        dim1 = aa.size(1)
        assert aa.size(2) == 3
        aa = aa.view(dim0*dim1, 3)
        rotmat = gut.angle_axis_to_rotation_matrix(aa)
        rotmat = rotmat.view(dim0, dim1, 4, 4)
        rotmat = rotmat[:, :, :3, :3]
    return rotmat


def angle_axis_to_rotation_matrix(angle_axis):
    aa = angle_axis
    if isinstance(aa, torch.Tensor):
        return __angle_axis_to_rotation_matrix_torch(aa)
    else:
        assert isinstance(aa, np.ndarray)
        aa_torch = torch.from_numpy(aa)
        rotmat_torch = __angle_axis_to_rotation_matrix_torch(aa_torch)
        return rotmat_torch.numpy()

def __rotation_matrix_to_angle_axis_torch(rotmat):
    rotmat = rotmat.clone()
    if rotmat.dim() == 2:
        assert rotmat.size(0) == 3
        assert rotmat.size(1) == 3
        rotmat0 = torch.zeros((1, 3, 4))
        rotmat0[0, :, :3] = rotmat
        rotmat0[:, 2, 3] = 1.0
        aa = gut.rotation_matrix_to_angle_axis(rotmat0)[0]
    elif rotmat.dim() == 3:
        dim0 = rotmat.size(0)
        assert rotmat.size(1) == 3
        assert rotmat.size(2) == 3
        rotmat0 = torch.zeros((dim0, 3, 4))
        rotmat0[:, :, :3] = rotmat
        rotmat0[:, 2, 3] = 1.0
        aa = gut.rotation_matrix_to_angle_axis(rotmat0)
    else:
        assert rotmat.dim() == 4
        dim0 = rotmat.size(0)
        dim1 = rotmat.size(1)
        assert rotmat.size(2) == 3
        assert rotmat.size(3) == 3
        rotmat0 = torch.zeros((dim0*dim1, 3, 4))
        rotmat0[:, :, :3] = rotmat.view(dim0*dim1, 3, 3)
        rotmat0[:, 2, 3] = 1.0
        aa = gut.rotation_matrix_to_angle_axis(rotmat0)
        aa = aa.view(dim0, dim1, 3)
    return aa


def rotation_matrix_to_angle_axis(rotmat):
    if isinstance(rotmat, torch.Tensor):
        return __rotation_matrix_to_angle_axis_torch(rotmat)
    else:
        assert isinstance(rotmat, np.ndarray)
        rotmat_torch = torch.from_numpy(rotmat)
        aa_torch = __rotation_matrix_to_angle_axis_torch(rotmat_torch)
        return aa_torch.numpy()

