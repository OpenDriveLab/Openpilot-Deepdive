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


CKPT_PATH = 'vis/M5_epoch_94.pth'  # Path to your checkpoint

# You can generate your own comma2k19_demo.txt to make some fancy demos
# val = Comma2k19SequenceDataset('data/comma2k19_demo.txt', 'data/comma2k19/','demo', use_memcache=False, return_origin=True)
val = Comma2k19SequenceDataset('data/comma2k19_val_non_overlap.txt', 'data/comma2k19/','demo', use_memcache=False, return_origin=True)
val_loader = DataLoader(val, 1, num_workers=0, shuffle=False)

planning_v0 = SequenceBaselineV1(5, 33, 1.0, 0.0, 'adamw')
planning_v0.load_state_dict(torch.load(CKPT_PATH))
planning_v0.eval().cuda()

seq_idx = 0
for b_idx, batch in enumerate(val_loader):
    os.mkdir('vis/M5_DEMO_%04d' % seq_idx)
    seq_inputs, seq_labels = batch['seq_input_img'].cuda(), batch['seq_future_poses'].cuda()
    origin_imgs = batch['origin_imgs']
    # camera_rotation_matrix_inv=batch['camera_rotation_matrix_inv'].numpy()[0]
    # camera_translation_inv=batch['camera_translation_inv'].numpy()[0]
    # camera_intrinsic=batch['camera_intrinsic'].numpy()[0]
    bs = seq_labels.size(0)
    seq_length = seq_labels.size(1)
    
    hidden = torch.zeros((2, bs, 512), device=seq_inputs.device)
    
    img_idx = 0
    for t in tqdm(range(seq_length)):

        with torch.no_grad():
            inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
            pred_cls, pred_trajectory, hidden = planning_v0(inputs, hidden)

            pred_conf = softmax(pred_cls, dim=-1).cpu().numpy()[0]
            pred_trajectory = pred_trajectory.reshape(planning_v0.M, planning_v0.num_pts, 3).cpu().numpy()

        inputs, labels = inputs.cpu(), labels.cpu()
        vis_img = (inputs.permute(0, 2, 3, 1)[0] * torch.tensor((0.2172, 0.2141, 0.2209, 0.2172, 0.2141, 0.2209)) + torch.tensor((0.3890, 0.3937, 0.3851, 0.3890, 0.3937, 0.3851)) )  * 255
        # print(vis_img.max(), vis_img.min(), vis_img.mean())
        vis_img = vis_img.clamp(0, 255)
        img_0, img_1 = vis_img[..., :3].numpy().astype(np.uint8), vis_img[..., 3:].numpy().astype(np.uint8)

        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
        # fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        # fig = plt.figure(figsize=(12, 9.9))  # W, H
        fig = plt.figure(figsize=(12, 9))  # W, H
        spec = fig.add_gridspec(3, 3)  # H, W
        ax1 = fig.add_subplot(spec[ 2,  0])  # H, W
        ax2 = fig.add_subplot(spec[ 2,  1])
        ax3 = fig.add_subplot(spec[ :,  2])
        ax4 = fig.add_subplot(spec[0:2, 0:2])

        ax1.imshow(img_0)
        ax1.set_title('network input [previous]')
        ax1.axis('off')

        ax2.imshow(img_1)
        ax2.set_title('network input [current]')
        ax2.axis('off')

        current_metric = (((pred_trajectory[pred_conf.argmax()] - labels.numpy()) ** 2).sum(-1) ** 0.5).mean().item()

        trajectories = list(pred_trajectory) + list(labels)
        confs = list(pred_conf) + [1, ]
        ax3 = draw_trajectory_on_ax(ax3, trajectories, confs, ylim=(0, 200))
        ax3.set_title('Mean L2: %.2f' % current_metric)
        ax3.grid()

        origin_img = origin_imgs[0, t, :, :, :].numpy()
        overlay = origin_img.copy()
        draw_path(pred_trajectory[pred_conf.argmax(), :], overlay, width=1, height=1.2, fill_color=(255,255,255), line_color=(0,255,0))
        origin_img = 0.5 * origin_img + 0.5 * overlay
        draw_path(pred_trajectory[pred_conf.argmax(), :], origin_img, width=1, height=1.2, fill_color=None, line_color=(0,255,0))

        ax4.imshow(origin_img.astype(np.uint8))
        ax4.set_title('project on current frame')
        ax4.axis('off')

        # ax4.imshow(img_1)
        # pred_mask = np.argmax(pred_conf)
        # pred_trajectory = [pred_trajectory[pred_mask, ...], ] + [batch['future_poses'].numpy()[0], ]
        # pred_conf = [pred_conf[pred_mask], ] + [1, ]
        # for pred_trajectory_single, pred_conf_single in zip(pred_trajectory, pred_conf):
        #     location = list((p + camera_translation_inv for p in pred_trajectory_single))
        #     proj_trajectory = np.array(list((camera_intrinsic @ (camera_rotation_matrix_inv @ l) for l in location)))
        #     proj_trajectory /= proj_trajectory[..., 2:3].repeat(3, -1)
        #     proj_trajectory /= 2
        #     proj_trajectory = proj_trajectory[(proj_trajectory[..., 0] > 0) & (proj_trajectory[..., 0] < 800)]
        #     proj_trajectory = proj_trajectory[(proj_trajectory[..., 1] > 0) & (proj_trajectory[..., 1] < 450)]
        #     ax4.plot(proj_trajectory[:, 0], proj_trajectory[:, 1], 'o-', label='gt' if pred_conf_single == 1.0 else 'pred - conf %.3f' % pred_conf_single, alpha=np.clip(pred_conf_single, 0.1, np.Inf))

        # ax4.legend()
        plt.tight_layout()
        plt.savefig('vis/M5_DEMO_%04d/%08d.jpg' % (seq_idx, img_idx), pad_inches=0.2, bbox_inches='tight')
        img_idx += 1
        # plt.show()
        plt.close(fig)

    seq_idx += 1
