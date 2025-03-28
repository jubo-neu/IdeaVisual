import torch
from kornia import create_meshgrid


def project_and_normalize(ref_grid, src_proj, length):
    src_grid = src_proj[:, :3, :3] @ ref_grid + src_proj[:, :3, 3:]
    div_val = src_grid[:, -1:]
    div_val[div_val<1e-4] = 1e-4
    src_grid = src_grid[:, :2] / div_val
    src_grid[:, 0] = src_grid[:, 0]/((length - 1) / 2) - 1
    src_grid[:, 1] = src_grid[:, 1]/((length - 1) / 2) - 1
    src_grid = src_grid.permute(0, 2, 1)
    return src_grid


def construct_project_matrix(x_ratio, y_ratio, Ks, poses):
    rfn = Ks.shape[0]
    scale_m = torch.tensor([x_ratio, y_ratio, 1.0], dtype=torch.float32, device=Ks.device)
    scale_m = torch.diag(scale_m)
    ref_prj = scale_m[None, :, :] @ Ks @ poses
    pad_vals = torch.zeros([rfn, 1, 4], dtype=torch.float32, device=ref_prj.device)
    pad_vals[:, :, 3] = 1.0
    ref_prj = torch.cat([ref_prj, pad_vals], 1)
    return ref_prj


def near_far_from_unit_sphere_using_camera_poses(camera_poses):
    R_w2c = camera_poses[..., :3, :3]
    t_w2c = camera_poses[..., :3, 3:]
    camera_origin = -R_w2c.permute(0,2,1) @ t_w2c
    camera_orient = R_w2c.permute(0,2,1)[...,:3,2:3]
    camera_origin, camera_orient = camera_origin[...,0], camera_orient[..., 0]
    a = torch.sum(camera_orient ** 2, dim=-1, keepdim=True)
    b = -torch.sum(camera_orient * camera_origin, dim=-1, keepdim=True)
    mid = b / a
    near, far = mid - 1.0, mid + 1.0
    return near, far
