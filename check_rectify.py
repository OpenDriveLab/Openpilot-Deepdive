import cv2
import torch
import numpy as np
from tqdm import tqdm

from data import PlanningDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F

from numba import jit
import time


@jit(nopython=True)
def projection(intrinsic, extrinsic, virtual_in, virtual_ex, h, w):
    point = np.array((w, h, 1.0), dtype=np.float32).reshape((3, 1))

    intrinsic_pad = np.zeros((3, 4), dtype=np.float32)
    intrinsic_pad[:3, :3] = virtual_in

    foo = intrinsic_pad @ virtual_ex
    foo = foo @ np.linalg.inv(extrinsic)

    temp = np.ones((4, 1), dtype=np.float32)
    temp[:3] = np.linalg.inv(intrinsic) @ point
    foo = foo @ temp
    foo = foo[:2]
    return foo.flatten()[::-1]  # h, w

@jit(nopython=True)
def gen_map_matrix(in_matrix, ex_matrix, virtual_in_mat, virtual_ex_mat, img_h, img_w):
    map_matrix = np.zeros((img_h, img_w, 2), dtype=np.float32)

    for h in range(img_h):
        for w in range(img_w):
            map_matrix[h][w] = projection(in_matrix, ex_matrix, virtual_in_mat, virtual_ex_mat, h, w)

    return map_matrix


val = PlanningDataset(split='val', json_path_pattern='p3_10pts_%s.json')
val_loader = DataLoader(val, 1, num_workers=1, shuffle=True)

intrinsic_matrices = []

virtual_in_mat = np.array(
    [
        [1266.4, 0, 800],
        [0, 1266.4, 450],
        [0, 0, 1]
    ],
    dtype=np.float32
)

virtual_ex_mat = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -1, 1.5],
        [1, 0, 0, -1.7],
        [0, 0, 0, 1]
    ],
    dtype=np.float32
)

extrinsic_matrices = []

for idx, sample in enumerate(tqdm(val_loader)):
    t1 = time.time()
    vis_img = (sample['input_img'].permute(0, 2, 3, 1)[0] + torch.tensor((0.3890, 0.3937, 0.3851, 0.3890, 0.3937, 0.3851)) ) * torch.tensor((0.2172, 0.2141, 0.2209, 0.2172, 0.2141, 0.2209)) * 255
    vis_img = vis_img.clamp(0, 255)
    img_0, img_1 = vis_img[..., :3].numpy().astype(np.uint8), vis_img[..., 3:].numpy().astype(np.uint8)


    in_matrix = sample['camera_intrinsic'][0].numpy().astype(np.float32)
    ex_matrix = sample['camera_extrinsic'][0].numpy().astype(np.float32)

    # in_flag = False
    # for m in extrinsic_matrices:
    #     if np.allclose(ex_matrix, m):
    #         in_flag = True
    #         continue
    # if in_flag:
    #     continue
    # extrinsic_matrices.append(ex_matrix)

    img_h, img_w = img_0.shape[:2]

    t2 = time.time()
    map_matrix = gen_map_matrix(in_matrix, ex_matrix, virtual_in_mat, virtual_ex_mat, img_h, img_w)
    t3 = time.time()

    map_matrix = map_matrix.astype(np.int)
    map_matrix[..., 0] = map_matrix[..., 0].clip(0, img_h-1)
    map_matrix[..., 1] = map_matrix[..., 1].clip(0, img_w-1)

    canvas = np.zeros_like(img_1)

    # for h in range(img_h):
    #     for w in range(img_w):
    #         canvas[h, w, :] = img_1[map_matrix[h, w, 0], map_matrix[h, w, 1], :]

    grid_indexes = map_matrix.reshape(-1, 2)
    grid_result = img_1[grid_indexes[:, 0], grid_indexes[:, 1]]  # img_h * img_w, 3
    grid_result = grid_result.reshape((img_h, img_w, 3))
    t4 = time.time()

    # cv2.imshow('vis_rect/original_%d.png' % idx, img_1[..., ::-1])
    # cv2.imshow('vis_rect/canvas_%d.png' % idx, canvas[..., ::-1])
    # cv2.imshow('vis_rect/grid_result_%d.png' % idx, grid_result[..., ::-1])
    # cv2.waitKey(0)
    print(t2 - t1, t3 - t2, t4 - t3)
