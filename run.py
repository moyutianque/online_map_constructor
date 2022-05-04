import gzip
import json
from config_parser import get_args
from collections import defaultdict
from map_constructor import Full_2D_Map_Countructor
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)
import random

def main():
    args = get_args()

    # 1. Processing VLNCE annotation files
    annt_file_path = args.annt_file.format(split=args.split)
    annt_gt_actions_path = args.annt_file_gt_actions.format(split=args.split)
    with gzip.open(annt_file_path, 'r') as f:
        eps_data = json.load(f)["episodes"]
    with gzip.open(annt_gt_actions_path, 'r') as f:
        gt_actions_data = json.load(f)
    
    data_dict = defaultdict(dict)
    for v in eps_data:
        scene_id = v['scene_id'].split('/')[-1].split('.')[0]

        data_dict[scene_id][v['episode_id']] = {
            'trajectory_id': v['trajectory_id'],
            'scene_id': scene_id,
            'start_position': v['start_position'],
            'start_rotation': v['start_rotation'],
            'instruction': v['instruction']['instruction_text'],
            'gt_actions': gt_actions_data[str(v['episode_id'])]['actions'],
            'gt_locations': gt_actions_data[str(v['episode_id'])]['locations'],
        }
    
    # 2. Process scenes:
    for k,v in data_dict.items():
        if k != 'r1Q1Z4BcV1o':
            continue
        print('\033[92m'+f'Swtich to scene {k}'+'\033[0m')
        full_map_constructor = Full_2D_Map_Countructor(scene_id=k, args = args)

        ep_ids = list(v.keys())
        random.shuffle(ep_ids)
        for i, (ep_id) in enumerate(tqdm(ep_ids)):
            ep_data = v[ep_id]
            full_map_constructor.parse_action_seq(
                ep_data['gt_actions'], start_sim_loc=ep_data['start_position'], 
                start_sim_rot=ep_data['start_rotation'], info={'idx':i, 'ep_id': ep_id, 'instruction': ep_data['instruction']},
                vis_step=args.vis_stepwise
            )
            
        # import ipdb;ipdb.set_trace() # breakpoint 43
        full_map_constructor.visualize(args.dump_location)
        full_map_constructor.sim.close()
        # import ipdb;ipdb.set_trace() # breakpoint 46
    

if __name__=='__main__':
    main()