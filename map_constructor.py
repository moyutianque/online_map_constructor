import torch
from torchvision import transforms
import os
import os.path as osp
from PIL import Image
import utils.pose as pu
import numpy as np
from utils.map_utils import get_local_map_boundaries
import habitat_sim
from habitat_sim import Simulator
from sim_config.config import make_cfg
import quaternion
from models.semantic_prediction import SemanticPredMaskRCNN
from utils.local_map_2D_constructor import Local_2D_Map_Constructor
from utils.visualizer import draw_map, draw_observation
import warnings

action_dict={
    0: "stop",
    1: "move_forward",
    2: "turn_left",
    3: "turn_right"
}

class Full_2D_Map_Countructor(object):
    def __init__(self, scene_id, args):
        super().__init__()
        self.scene_id = scene_id
        self.args = args

        # TODO: [STEP 1] init global map container
        self.gmap_init()

        # [STEP 2] Build simulator env
        scene_file_glb = args.scene_path_template.format(scene=scene_id)
        settings=dict() # can update settings here
        self.cfg = make_cfg(scene_file_glb, settings)
        self.sim = Simulator(self.cfg)

        # [STEP 3] Init visual perception model (Mask RCNN)
        self.res = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.frame_height, args.frame_width), interpolation=Image.NEAREST)
        ])
        self.sem_pred = SemanticPredMaskRCNN(args)

        # TODO: Build local map constructor
        self.local_map_constructor = Local_2D_Map_Constructor(args)

    def parse_action_seq(self, action_seq, start_sim_loc, start_sim_rot, info, vis_step=False):
        start_pos = start_sim_loc
        rotation = np.array([start_sim_rot[3], *start_sim_rot[:3]]) # RxR specific
        start_rot = quaternion.from_float_array(rotation)

        # Initialize new agent position
        agent = self.sim.initialize_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(start_pos)  # world space
        agent_state.rotation = start_rot
        agent.set_state(agent_state)

        if not hasattr(self, 'last_sim_location'):
            self.last_sim_location = self.get_sim_location() 
            if info['idx'] == 0:
                # Zehao @ May 2: Unify the coordinate according to real sim location
                scene = self.sim.semantic_scene
                # lower_bound, upper_bound = scene.aabb.center - (scene.aabb.sizes/2), scene.aabb.center + (scene.aabb.sizes/2)
                # xs, zs, ys = lower_bound
                # self.origins.fill(0)
                # self.last_sim_location = [xs, ys, 0]
                ys, zs, xs = scene.aabb.center 
                self.last_sim_location = [-xs, -ys, -np.pi/2] # x y o # assign initial center to map center
                
        for i, action_index in enumerate(action_seq):
            action = action_dict[action_index]
            if action == 'stop':
                return

            obs = self.sim.step(action)
            # ==> obs['color_sensor'] uint8 RGBD image, with shape [width, height, 4]
            # ==> obs['depth_sensor'] float32 Depth image, with shape [width, height] (distance in meter)
            # ==> obs['collided] bool value for wether collided with obstacle.
            if obs['collided'] == True:
                warnings.warn(f"[Collision Check] Check ep {info['ep_id']} in scene {self.scene_id} with step {i}", FutureWarning)

            state = self._preprocess_obs(obs)

            dx, dy, do = self.get_pose_change()
            pose_obs = [dx, dy, do]
            pose_obs = torch.tensor([pose_obs]).float().to(self.args.device) # batch the pose change
            _, self.local_map, _, self.local_pose = self.local_map_constructor( state, pose_obs, self.local_map, self.local_pose)
            self.update_full_map()

            if vis_step:
                self.visualize(self.args.dump_location, ep_id=info['ep_id'], step=i, e=0)
                if os.environ.get('DEBUG', False): 
                    draw_observation(state[0, 0:3,:,:], state[0,3,:,:])
                    print(info['instruction'])

    def visualize(self, dump_dir, ep_id=0, step=None, e=0):
        stepinfo_and_agent_pose=None
        if step is not None:
            stepinfo_and_agent_pose = (
                (ep_id, step),
                self.planner_pose_inputs[e]
            )
        draw_map(
            self.full_map, dump_dir, 
            self.args.map_resolution, self.scene_id, 
            stepinfo_and_agent_pose=stepinfo_and_agent_pose, 
            e=0, out_map_size=480
        )
        
    def _preprocess_obs(self, obs):
        rgb = obs['color_sensor'][:,:,:3]
        depth = obs['depth_sensor']
        
        sem_seg_pred = self._get_sem_pred(rgb.astype(np.uint8), use_seg=True)
        
        depth = self._preprocess_depth(depth, self.args.min_depth, self.args.max_depth)
        ds = self.args.env_frame_width // self.args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]
        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)
        state = torch.from_numpy(state[None, ...]).float().to(self.args.device)
        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        # Clip and normalize depth observation as HabitatSimDepthSensor
        depth = np.clip(depth, min_d, max_d)
        depth = (depth -  min_d) / (
                max_d - min_d
        ) * 1

        # Other process
        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth

    def _get_sem_pred(self, rgb, use_seg=True):
        if use_seg:
            semantic_pred, _ = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            # self.rgb_vis = rgb[:, :, ::-1]
        return semantic_pred
    
    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""
        agent_state = self.sim.get_agent(0).get_state()
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do

    def gmap_init(self):
        args = self.args
        num_scenes=1
        # Initialize map variables:
        # Full map consists of multiple channels containing the following:
        # 1. Obstacle Map
        # 2. Exploread Area
        # 3. Current Agent Location
        # 4. Past Agent Locations
        # 5,6,7,.. : Semantic Categories
        nc = args.num_sem_categories + 4  # num channels
        map_size = args.map_size_cm // args.map_resolution
        full_w, full_h = map_size, map_size
        local_w = int(full_w / args.global_downscaling)
        local_h = int(full_h / args.global_downscaling)
        full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(args.device)
        local_map = torch.zeros(num_scenes, nc, local_w,
                                local_h).float().to(args.device)
        # Initial full and local pose
        full_pose = torch.zeros(num_scenes, 3).float().to(args.device) # x,y,o in degree, but last_sim_location is in radian
        local_pose = torch.zeros(num_scenes, 3).float().to(args.device)

        # Origin of local map
        origins = np.zeros((num_scenes, 3))

        # Local Map Boundaries
        lmb = np.zeros((num_scenes, 4)).astype(int)

        # Planner pose inputs has 7 dimensions 1-3 store continuous global agent location 4-7 store local map boundaries
        planner_pose_inputs = np.zeros((num_scenes, 7))
        def init_map_and_pose_for_env(e):
            full_map[e].fill_(0.)
            full_pose[e].fill_(0.)
            full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

            locs = full_pose[e].cpu().numpy()
            planner_pose_inputs[e, :3] = locs
            r, c = locs[1], locs[0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c), (local_w, local_h), (full_w, full_h), 
                        global_downscaling= self.args.global_downscaling)

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [
                lmb[e][2] * args.map_resolution / 100.0,
                lmb[e][0] * args.map_resolution / 100.0, 
                0.0
            ]

            local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - torch.from_numpy(origins[e]).to(args.device).float()
        

        init_map_and_pose_for_env(0)
        self.full_map = full_map
        self.lmb = lmb
        self.local_map = local_map
        self.planner_pose_inputs = planner_pose_inputs
        self.origins = origins
        self.local_w = local_w
        self.local_h = local_h
        self.full_w = full_w
        self.full_h = full_h
        self.full_pose = full_pose
        self.local_pose = local_pose
    
    def update_full_map(self, e=0):
        # Use local_map and local_pose to update full map
        locs = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs + self.origins
        self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel

        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]
    
        # Update visited map and current location channel
        self.local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.

        # Assign geocentric local map to full map
        self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]] = \
                    self.local_map[e]
        
        # Switch back to global coordinate
        self.full_pose[e] = self.local_pose[e] + \
            torch.from_numpy(self.origins[e]).to(self.args.device).float()
        # Get new boundary around agent new location
        locs = self.full_pose[e].cpu().numpy()
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]
        self.lmb[e] = get_local_map_boundaries((loc_r, loc_c), (self.local_w, self.local_h), (self.full_w, self.full_h), 
                    global_downscaling= self.args.global_downscaling)

        self.planner_pose_inputs[e, 3:] = self.lmb[e]
        self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
                        self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]
        # Crop new local map with agent in the center
        self.local_map[e] = self.full_map[e, :,
                                self.lmb[e, 0]:self.lmb[e, 1],
                                self.lmb[e, 2]:self.lmb[e, 3]]
        self.local_pose[e] = self.full_pose[e] - \
            torch.from_numpy(self.origins[e]).to(self.args.device).float() 
   