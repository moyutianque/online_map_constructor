import torch
import torch.nn as nn
from torch.nn import functional as F
import utils.depth_utils as du
import numpy as np
from utils.map_utils import get_grid

class Local_2D_Map_Constructor(nn.Module):
    """
    A modified version of original Semantic_Mapping class
    """
    def __init__(self, args):
        """
        The coordinate used in this module is XYZ: xy for plane, z for height
        Args:
            device: tensor device
            frame_height / frame_width: RGBD observation height and width
            map_size_cm: map width and height size in cm
            map_resolution: number of times to downscale the real-world centimeters (cm)
            global_downscaling: a global map resolution downscale (a further downscale)
            vision_range: the actual local map plane region in front of agent
            hfov: Horizontal Field of View in cm; camera matrix components
            camera_height: agent's view height in meters
            du_scale: the scale to be considered in depth image while generating points (how many pixels to skip)

            exp_pred_threshold: voxel threshold, in obstacle channel of voxel map, upper bound of obstacle height at each position that cannot be stand on. e.g. threashold 5 means if one position has number of voxels occupied greater than 5, this position is completely not possible to stand on
            map_pred_threshold: similar to exp_pred_threshold, but consider a small range around agent height. exp_pred_threshold is defined on entire height range
            cat_pred_threshold: similar to above

            num_sem_categories: number of semantic categories in total
            num_processes: batch_size, we default it to 1
        """
        super().__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution 
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        
        # TODO: 
        # NOTE by zehao:
        self.shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]

        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        vr = self.vision_range

        # Grid map to accomadate voxel values (each represent occupied or not occupied)
        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        # record of  (1 channel) + downscaled semantic segmentations (16 channel)
        self.feat = torch.ones(
            args.num_processes, 1 + self.num_sem_categories,
            (self.screen_h // self.du_scale) * (self.screen_w // self.du_scale)
        ).float().to(self.device)

    def forward(self, obs, pose_obs, maps_last, poses_last):
        """
        Args:
            obs: current observation RGBD (4 channel) + semantic segmentations (16 channel)
            pose_obs: current GT pose changes (no noise)
            maps_last: 
            poses_last: last GT pose (no noise)
        Return:
            fp_map_pred:
            map_pred:
            pose_pred:
            current_poses:
        """
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]

        # MAP 1. Depth image to point cloud (egocentric coordinates)
        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)
        # Rectify camera elevation (default is horizontal camera, no elevation)
        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, self.device)
        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)

        # MAP 2. Generate egocentric voxels
        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float() # every point in current observation has being assigned a 3d coordinate
        # 2.1 Standardize point cloud coordinates to [0,1], NOTE: still does not deal with points out of vision_range 
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:, :, :]
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)
       
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)

        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        # 2.2 based on segmentation and point clouds, construct egocentric voxels
        # TODO: understand details of voxel calculation
        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)
        # ==> egocentric voxels (bs, self.feat.shape[1], vision_range, vision_range, downscaled_height)
        # TODO: it seems splat_feat_nd give the first feat dimension, i.e. voxels[:,0:1] 0/1 value represent wether this height is occupied

        # MAP 3. Define observed area (map_pred) and explorable area (exp_pred) by number of occupied voxel at each position
        min_z = int(25 / z_resolution - min_h) 
        max_z = int((self.agent_height + 1) / z_resolution - min_h)
        agent_height_proj = voxels[..., min_z:max_z].sum(4) # the height sum in the voxel area from height 25cm to agent_height+1
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        # MAP 4. Assign thresholded obstacle map and explorable map and semantic map to agent view container
        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        #### shift to agent cetric, which fit the relative coord of voxels (vision_range//2, 0, height//2) is camera
        #   [   ]     [   ]
        #  ^      =>    ^
        #  |            |
        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        # TODO: Why not use all_height_proj, it seems agent_height_proj will miss small object on the ground or on the ceiling
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        # POSE 1. Following Neural SLAM paper, thie pose_pred module is predicted (dx,dy,do), it will be accumulated to observed pose to get the corrected_pose 
        pose_pred = poses_last
        # pose_pred = None

        # POSE 2. Pose correction module (we assume no noise in pose sensor but left the step here)
        corrected_pose = pose_obs

        # POSE 3. Calculate current pose in meters based on simulator pose change (last coord can be map coord in the same scale) 
        #         (a procedure that assume pose change is not from simulator, so should estimate current pose only by last pose and pose change) 
        def get_new_pose_batch(pose, rel_pose_change):
            # 180 / pi = 57.29577951308232
            o_in_radians = pose[:, 2] / 57.29577951308232
            pose[:, 1] += rel_pose_change[:, 0] * torch.sin(o_in_radians) \
                + rel_pose_change[:, 1] * torch.cos(o_in_radians)
            pose[:, 0] += rel_pose_change[:, 0] * torch.cos(o_in_radians) \
                - rel_pose_change[:, 1] * torch.sin(o_in_radians)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose
        current_poses = get_new_pose_batch(poses_last, corrected_pose)

        # POSE 4. TODO: do not know why shift current position and orientation, might be a not unified transfer between position and rotation coordinate while dealing with local map alignment to global map
        #         The rotation matrix implementated in get_grid is counter-clockwise rotation
        #  NOTE: st means Spatial Transformation
        st_pose = current_poses.clone().detach()
        st_pose[:, :2] = - (st_pose[:, :2] * 100.0 / self.resolution - self.map_size_cm // (self.resolution * 2)) \
                         / (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2]) # Unknown reason

        # pytorch implementation of affine transformation
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device) 
        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)
        
        # Join old maps and new maps 
        # TODO: consider 3D consistency here
        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1) 
        map_pred, _ = torch.max(maps2, 1)
        return fp_map_pred, map_pred, pose_pred, current_poses