import numpy as np
from constants import color_palette
from PIL import Image
import cv2
import os
import os.path as osp
import time
import torch

def get_contour_points(pos, origin, size=10):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])

def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat

def draw_observation(rgb, depth):
    # return
    # import ipdb;ipdb.set_trace() # breakpoint 30
    os.makedirs('./tmp', exist_ok=True)
    rgb_img = Image.fromarray(np.uint8(np.transpose(rgb.cpu().detach().numpy(), (1,2,0)))).convert('RGB')
    print(torch.min(depth), torch.max(depth))
    depth_img = Image.fromarray((depth.cpu().detach().numpy() / 10 * 255).astype(np.uint8), mode="L")
    tc = time.time()
    rgb_img.save(f'./tmp/obs-{tc}_rgb.jpg')
    depth_img.save(f'./tmp/obs-{tc}_depth.jpg')

def draw_map(full_maps_raw, dump_dir, map_resolution, scene_id, stepinfo_and_agent_pose=None, e=0, out_map_size=480):
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(osp.join(dump_dir, scene_id), exist_ok=True)
    full_maps = full_maps_raw.cpu().detach()

    if os.environ.get('DEBUG', False): 
        pil_img = Image.fromarray(np.uint8(full_maps[e,0,:,:] * 255), 'L')
        pil_img.save(f'./tmp/map-{stepinfo_and_agent_pose[0][1]}.png')
        return

    map_pred = full_maps[e, 0, :, :].numpy()
    exp_pred = full_maps[e, 1, :, :].numpy()
    visited_vis = full_maps[e, 3, :, :]

    sem_map = full_maps[e, 4:, :, :] 

    sem_map[-1, :, :] = 1e-5 # NOTE: [IMPORTANT] add no-cat value

    sem_map = sem_map.argmax(0).numpy()
    sem_map += 5 

    no_cat_mask = sem_map == 20

    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    
    # vis_mask = visited_vis[gx1:gx2, gy1:gy2] == 1 # NOTE: can remove visited path here
    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    # sem_map[vis_mask] = 3
    
    # Start drawing semantic map
    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (out_map_size, out_map_size),
                                interpolation=cv2.INTER_NEAREST)
                                
    if stepinfo_and_agent_pose is not None:
        step_info = stepinfo_and_agent_pose[0]
        agent_pose = stepinfo_and_agent_pose[1]
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent_pose
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        # pos = (
        #     (start_x * 100. / map_resolution - gy1) * 480 / map_pred.shape[0],
        #     (map_pred.shape[1] - start_y * 100. / map_resolution + gx1) * 480 / map_pred.shape[1],
        #     np.deg2rad(-start_o)
        # )
        pos = (
            (start_x * 100. / map_resolution) * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / map_resolution) * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = get_contour_points(pos, origin=(0, 0))
        color = (int(color_palette[11] * 255),
                    int(color_palette[10] * 255),
                    int(color_palette[9] * 255))
        
        cv2.drawContours(sem_map_vis, [agent_arrow], 0, color, -1)
    
        fn = osp.join(dump_dir, scene_id, f"ep{step_info[0]}-step{step_info[1]}.png")
        cv2.imwrite(fn, sem_map_vis)
    else:
        fn = osp.join(dump_dir, scene_id, f"__fullmap.png")
        cv2.imwrite(fn, sem_map_vis)



