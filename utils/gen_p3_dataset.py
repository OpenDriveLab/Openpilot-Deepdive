import json
import random
random.seed(0)
import numpy as np

from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation


# Hyper-Params
DATA_ROOT = 'data/nuscenes'
SPLIT = 'v1.0-trainval'
NUM_RGB_IMGS = 2
NUM_FUTURE_TRAJECTORY_PTS = 20
OUTPUT_JSON_NAME = 'data/p3_%s.json'


def get_samples(nusc, scenes):
    samples = []
    # list of dicts, where
    # 'img': LIST of filenames 0, 1, ..., NUM_RGB_IMGS - 1. 
    # NUM_RGB_IMGS - 1 is the frame of 'current' timestamp
    # 'pt_%d': LIST of future points offset by current img 0, 1, ..., NUM_FUTURE_TRAJECTORY_PTS - 1
    # 0 is the point of the 'very next' timestamp
    for scene in scenes:
        assert len(scene) >= NUM_RGB_IMGS + NUM_FUTURE_TRAJECTORY_PTS
        valid_start_tokens = scene[NUM_RGB_IMGS-1 : -NUM_FUTURE_TRAJECTORY_PTS]

        for idx, cur_token in enumerate(valid_start_tokens):
            img_tokens = scene[idx:idx+NUM_RGB_IMGS]
            point_tokens = scene[idx+NUM_RGB_IMGS:idx+NUM_RGB_IMGS+NUM_FUTURE_TRAJECTORY_PTS]
            
            # Images
            imgs = list(nusc.get('sample_data', nusc.get('sample', token)['data']['CAM_FRONT'])['filename'] for token in img_tokens)
            
            cur_ego_pose = nusc.get('ego_pose', nusc.get('sample_data', nusc.get('sample', cur_token)['data']['CAM_FRONT'])['ego_pose_token'])
            
            ego_rotation_matrix = Rotation.from_quat(np.array(cur_ego_pose['rotation'])[[1,2,3,0]]).as_matrix()
            ego_tranlation = np.array(cur_ego_pose['translation'])
            ego_rotation_matrix_inv = np.linalg.inv(ego_rotation_matrix)
            ego_tranlation_inv = -ego_tranlation

            future_poses = list(nusc.get('ego_pose', nusc.get('sample_data', nusc.get('sample', token)['data']['CAM_FRONT'])['ego_pose_token'])['translation'] for token in point_tokens)
            future_poses = list(ego_rotation_matrix_inv @ (np.array(future_pose)+ego_tranlation_inv) for future_pose in future_poses)
            future_poses = list(list(p) for p in future_poses)  # for json

            samples.append(dict(imgs=imgs, future_poses=future_poses))

    return samples


# Load NuScenes dataset
nusc = NuScenes(version=SPLIT, dataroot=DATA_ROOT, verbose=True)

# get all scenes into time structure
all_scenes = []
for scene in nusc.scene:
    cur_token = scene['first_sample_token']
    cur_scene_tokens = []
    while cur_token != '':
        cur_scene_tokens.append(cur_token)
        cur_sample = nusc.get('sample', cur_token)
        cur_token = cur_sample['next']

    all_scenes.append(cur_scene_tokens)

random.shuffle(all_scenes)

length_all_scenes = len(all_scenes)
print('Altogether', length_all_scenes, 'scenes')

train_samples = get_samples(nusc, all_scenes[:int(length_all_scenes * 0.8)])
val_samples = get_samples(nusc, all_scenes[int(length_all_scenes * 0.8):])

json.dump(train_samples, open(OUTPUT_JSON_NAME % 'train', 'w'), indent='\t')
json.dump(val_samples, open(OUTPUT_JSON_NAME % 'val', 'w'), indent='\t')
