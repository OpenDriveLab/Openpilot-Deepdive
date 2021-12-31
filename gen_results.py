from utils import draw_path, draw_trajectory_on_ax
import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from torch.nn.functional import softmax

from data import PlanningDataset, Comma2k19SequenceDataset
from main import SequenceBaselineV1
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# val = Comma2k19SequenceDataset('data/comma2k19_demo.txt', 'data/comma2k19/','demo', use_memcache=False, return_origin=True)
val = Comma2k19SequenceDataset('data/comma2k19_val_non_overlap.txt', 'data/comma2k19/','demo', use_memcache=False, return_origin=False)
val_loader = DataLoader(val, 1, num_workers=1, shuffle=False, prefetch_factor=1)

planning_v0 = SequenceBaselineV1(5, 33, 1.0, 0.0, 'adamw')
planning_v0.load_state_dict(torch.load('vis/M5_epoch_94.pth'))
planning_v0.eval().cuda()

result_dict = dict(
    pred = np.zeros((402650, 5, 33, 3), dtype=np.float32),
    confidence = np.zeros((402650, 5), dtype=np.float32),
    gt = np.zeros((402650, 1, 33, 3), dtype=np.float32),
)

smp_idx = 0
with torch.no_grad():
    for b_idx, batch in enumerate(tqdm(val_loader)):
        seq_inputs, seq_labels = batch['seq_input_img'].cuda(), batch['seq_future_poses']
        bs = seq_labels.size(0)
        seq_length = seq_labels.size(1)
        
        hidden = torch.zeros((2, bs, 512), device=seq_inputs.device)
        
        for t in tqdm(range(seq_length), leave=False):

            inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
            pred_cls, pred_trajectory, hidden = planning_v0(inputs, hidden)

            pred_conf = softmax(pred_cls, dim=-1).cpu().numpy()[0]
            pred_trajectory = pred_trajectory.reshape(planning_v0.M, planning_v0.num_pts, 3).cpu().numpy()

            result_dict['pred'][smp_idx] = pred_trajectory
            result_dict['confidence'][smp_idx] = pred_conf
            result_dict['gt'][smp_idx] = labels

            smp_idx += 1

np.save('M5.npy', result_dict)
