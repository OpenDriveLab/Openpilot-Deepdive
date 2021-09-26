import torch
import numpy as np
from tqdm import tqdm

from data import PlanningDataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import cv2


val = PlanningDataset(split='val', json_path_pattern='p3_10pts_%s.json')
val_loader = DataLoader(val, 1, num_workers=0)

class JudgeDataset(PlanningDataset):
    def __init__(self, json_path_pattern, split):
        super().__init__(json_path_pattern=json_path_pattern, split=split)
        self.intrinsic_dict = {1252: 0, 1262: 1, 1266: 2}

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs = sample['imgs']

        img = Image.open(os.path.join(self.img_root, imgs[0]))
        intrinsic_type=int(sample['camera_intrinsic'][0][0])

        return dict(
            input_img=np.asarray(img),
            cls=self.intrinsic_dict[intrinsic_type]
        )

val_test = JudgeDataset(split='val', json_path_pattern='p3_10pts_%s.json')
val_loader_test = DataLoader(val_test, 1, num_workers=0)
stat = {0:0, 1:0, 2:0}

for idx, v in enumerate(tqdm(val_loader_test)):
    cls = v['cls']
    stat[cls.item()] += 1
    cv2.imwrite('vis/%d/%d.jpg' % (cls, idx), v['input_img'][0].numpy().astype(np.uint8)[..., ::-1])

print(stat)


intrinsic_matrices = []

for sample in tqdm(val_loader):
    cim = sample['camera_intrinsic']

    in_flag = False
    for m in intrinsic_matrices:
        if torch.allclose(cim, m):
            in_flag = True
            continue
    if in_flag:
        continue
    intrinsic_matrices.append(cim)
    print(len(intrinsic_matrices))
    for m in intrinsic_matrices:
        print(m[0][0][0])

print(intrinsic_matrices)


# [[1.2664e+03, 0.0000e+00, 8.1627e+02],
#  [0.0000e+00, 1.2664e+03, 4.9151e+02],
#  [0.0000e+00, 0.0000e+00, 1.0000e+00]]

# [[1.2528e+03, 0.0000e+00, 8.2659e+02],
#  [0.0000e+00, 1.2528e+03, 4.6998e+02],
#  [0.0000e+00, 0.0000e+00, 1.0000e+00]]

# [[1.2628e+03, 0.0000e+00, 7.8668e+02],
#  [0.0000e+00, 1.2628e+03, 4.3799e+02],
#  [0.0000e+00, 0.0000e+00, 1.0000e+00]]

extrinsic_matrices = []

for sample in tqdm(val_loader):
    cim = sample['camera_extrinsic']

    in_flag = False
    for m in extrinsic_matrices:
        if torch.allclose(cim, m):
            in_flag = True
            continue
    if in_flag:
        continue
    extrinsic_matrices.append(cim)
    print(len(extrinsic_matrices))

print(extrinsic_matrices)

# [[ 5.6848e-03, -9.9998e-01,  8.0507e-04, -1.7008e+00],
#  [-5.6367e-03, -8.3712e-04, -9.9998e-01, -1.5946e-02],
#  [ 9.9997e-01,  5.6801e-03, -5.6413e-03, -1.5110e+00],
#  [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]

# [[ 0.0103, -0.9999, -0.0122, -1.7220],
#  [ 0.0084,  0.0123, -0.9999, -0.0048],
#  [ 0.9999,  0.0102,  0.0086, -1.4949],
#  [ 0.0000,  0.0000,  0.0000,  1.0000]]

# [[-0.0047, -1.0000,  0.0056, -1.6710],
#  [ 0.0136, -0.0056, -0.9999,  0.0260],
#  [ 0.9999, -0.0046,  0.0136, -1.5360],
#  [ 0.0000,  0.0000,  0.0000,  1.0000]]
