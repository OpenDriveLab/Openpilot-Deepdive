import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
# from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


device_frame_from_view_frame = np.array([
    [ 0.,  0.,  1.],
    [ 1.,  0.,  0.],
    [ 0.,  1.,  0.]
])
view_frame_from_device_frame = device_frame_from_view_frame.T

# MED model
MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_YUV_SIZE = (MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1] * 3 // 2)
MEDMODEL_CY = 47.6

medmodel_fl = 910.0
medmodel_intrinsics = np.array([
    [medmodel_fl,  0.0,  0.5 * MEDMODEL_INPUT_SIZE[0]],
    [0.0,  medmodel_fl,                   MEDMODEL_CY],
    [0.0,  0.0,                                   1.0]])


def calibration(extrinsic_matrix, cam_intrinsics, device_frame_from_road_frame=None):
    if device_frame_from_road_frame is None:
        device_frame_from_road_frame = np.hstack((np.diag([1, -1, -1]), [[0], [0], [1.51]]))
    med_frame_from_ground = medmodel_intrinsics@view_frame_from_device_frame@device_frame_from_road_frame[:,(0,1,3)]
    ground_from_med_frame  = np.linalg.inv(med_frame_from_ground)


    extrinsic_matrix_eigen = extrinsic_matrix[:3]
    camera_frame_from_road_frame = np.dot(cam_intrinsics, extrinsic_matrix_eigen)
    camera_frame_from_ground = np.zeros((3,3))
    camera_frame_from_ground[:,0] =  camera_frame_from_road_frame[:,0]
    camera_frame_from_ground[:,1] =  camera_frame_from_road_frame[:,1]
    camera_frame_from_ground[:,2] =  camera_frame_from_road_frame[:,3]
    warp_matrix = np.dot(camera_frame_from_ground, ground_from_med_frame)

    return warp_matrix


if __name__ == '__main__':
    from data import PlanningDataset
    dataset = PlanningDataset(split='val')
    for idx, data in tqdm(enumerate(dataset)):
        imgs = data["input_img"]
        img0 = imgs[0]
        camera_rotation_matrix = np.linalg.inv(data["camera_rotation_matrix_inv"].numpy())
        camera_translation = -data["camera_translation_inv"].numpy()
        camera_extrinsic = np.vstack((np.hstack((camera_rotation_matrix, camera_translation.reshape((3, 1)))), np.array([0, 0, 0, 1])))
        camera_extrinsic = np.linalg.inv(camera_extrinsic)
        cv2.imshow("origin_img",img0)
        cv2.waitKey(0)
        warp_matrix = calibration(camera_extrinsic, data["camera_intrinsic"].numpy())
        transformed_img = cv2.warpPerspective(src = img0, M = warp_matrix, dsize= (512,256), flags= cv2.WARP_INVERSE_MAP)
        cv2.imshow("warped_img",transformed_img)
        cv2.waitKey(0)
