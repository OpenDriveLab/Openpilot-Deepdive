import glob
import os
import json
import copy

import cv2
from matplotlib import pyplot
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

import numpy as np
from tqdm import tqdm

from scipy.spatial.transform import Rotation

# class nuscene2sensetime:
#     input_path = ""
#     output_path = ""
#     def __init__(self, input_path, output_path):
#         self.input_path = input_path
#         self.output_path = output_path

#         pass
#     pass

def convert(version, input_path, output_path):
    if output_path[-1] == '/':
        output_path = output_path[:-1]
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    nusc = NuScenes(version=version, dataroot=input_path, verbose=True)

    for scene in tqdm(nusc.scene, desc="process"):
        first_sample_token = scene['first_sample_token']
        sample_token = first_sample_token
        output_path_scene = output_path +'/{}'.format(scene['token'])
        output_path_image = output_path_scene + '/images'
        if not os.path.exists(output_path_image):
            os.makedirs(output_path_image)
        output_path_lidar_object = output_path_scene + '/lidar_object'
        if not os.path.exists(output_path_lidar_object):
            os.makedirs(output_path_lidar_object)
        while sample_token != '':
            sample = nusc.get('sample', sample_token)

            sensor = 'CAM_FRONT'
            cam_front_data = nusc.get('sample_data', sample['data'][sensor])
            ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_rotation_matrix = Rotation.from_quat(np.array(ego_pose['rotation'])[[1,2,3,0]]).as_matrix()
            ego_tranlation = np.array(ego_pose['translation'])
            ego_rotation_matrix_inv = np.linalg.inv(ego_rotation_matrix)
            ego_tranlation_inv = -ego_tranlation
            ego_yaw = quaternion_yaw(Quaternion(ego_pose['rotation']))

            calibration_para = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
            camera_intrinsic = np.array(calibration_para['camera_intrinsic'])
            camera_rotation_matrix = Rotation.from_quat(np.array(calibration_para['rotation'])[[1,2,3,0]]).as_matrix()
            camera_translation = np.array(calibration_para['translation'])
            camera_rotation_matrix_inv = np.linalg.inv(camera_rotation_matrix)
            camera_translation_inv = -camera_translation
            camera_extrinsic = np.vstack((np.hstack((camera_rotation_matrix_inv, camera_translation_inv.reshape((3, 1)))), np.array([0, 0, 0, 1])))
            # print(calibration_para)

            para = dict()
            para["camera_intrinsic"] = np.hstack((camera_intrinsic, np.zeros(3).reshape(3, 1))).tolist()

            with open(output_path_scene + '/parameter.json', 'w') as jsonf:
                json.dump(para, jsonf)



            image = cv2.imread(input_path + '/' + cam_front_data['filename'])
            cv2.imwrite(output_path_image + '/{}.jpg'.format(sample['timestamp']), image)

            sample_token = sample['next']

            object_list = list()
            for anns_token in sample['anns']:
                object = dict()
                annotation = nusc.get('sample_annotation', anns_token)
                yaw = quaternion_yaw(Quaternion(annotation['rotation']))
                location = np.array(annotation['translation'])
                size = np.array([annotation['size'][0], annotation['size'][2], annotation['size'][1]])
                location = location + ego_tranlation_inv
                location = ego_rotation_matrix_inv @ location
                location = location + camera_translation_inv
                location = camera_rotation_matrix_inv @ location
                location = location.reshape(3)
                if location[2] <= 0:
                    continue
                object["Cam3Dpoints"] = location.tolist()
                object["id"] = annotation['instance_token']
                object["pry_ZYX"] = [-(yaw-ego_yaw), 0, 0]
                object["size"] = size.tolist()
                object["type"] = annotation["category_name"]
                object_list.append(copy.deepcopy(object))
            with open(output_path_lidar_object + '/{}.json'.format(sample["timestamp"]), 'w') as jsonf :
                json.dump(object_list, jsonf)

        # nusc.render_sample_data(cam_front_data['token'])

        # print(scene)



if __name__ == "__main__":
    convert('v1.0-trainval', '/data/4T_disk2/nuscenes/trainval', '/data/4T_disk2/nuscenes_sensetime')
