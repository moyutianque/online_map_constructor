import torch
from torch.nn import functional as F
import numpy as np

def get_grid(pose, grid_size, device):
    """
    Output affine grid (indicate the value of each target 2D output comes from which input location)
    Input:
        `pose` FloatTensor(bs, 3) -- (x, y, orientation)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w) target output tensor size
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180.
    cos_t = t.cos()
    sin_t = t.sin()
    
    # theta 1 is rotation matrix for 2D
    #  [[cos_t, -sin_t, 0],
    #   [sint_t, cos_t, 0]] 
    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    # theta 2 is translation matrix for 2D
    #  [[1, 0, x],
    #   [0, 1, y]]
    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    # NOTE by zehao: as described in pytorch document, the align_corners 
    #                option should keep the same as next step grid_sample
    rot_grid = F.affine_grid(theta1, torch.Size(grid_size), align_corners=True)
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size), align_corners=True)

    return rot_grid, trans_grid

def get_local_map_boundaries(agent_loc, local_sizes, full_sizes, global_downscaling):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

    return [gx1, gx2, gy1, gy2]

