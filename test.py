import torch
import numpy as np
from torch.nn.functional import softmax

from data import PlanningDataset
from main import PlanningBaselineV0
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

val = PlanningDataset(split='val')
val_loader = DataLoader(val, 1, num_workers=0, shuffle=False)

planning_v0 = PlanningBaselineV0.load_from_checkpoint('epoch=910-step=92921.ckpt', M=3, num_pts=20, mtp_alpha=1.0, lr=0)
planning_v0.eval().cuda()

for b_idx, batch in enumerate(val_loader):
    inputs, labels = batch['input_img'], batch['future_poses']
    camera_rotation_matrix_inv=batch['camera_rotation_matrix_inv'].numpy()[0]
    camera_translation_inv=batch['camera_translation_inv'].numpy()[0]
    camera_intrinsic=batch['camera_intrinsic'].numpy()[0]

    with torch.no_grad():
        pred_cls, pred_trajectory = planning_v0(inputs.cuda())
        pred_conf = softmax(pred_cls, dim=-1).cpu().numpy()[0]
        pred_trajectory = pred_trajectory.reshape(3, 20, 3).cpu().numpy()

    vis_img = (inputs.permute(0, 2, 3, 1)[0] + torch.tensor((0.3890, 0.3937, 0.3851, 0.3890, 0.3937, 0.3851)) ) * torch.tensor((0.2172, 0.2141, 0.2209, 0.2172, 0.2141, 0.2209)) * 255
    vis_img = vis_img.clamp(0, 255)
    img_0, img_1 = vis_img[..., :3].numpy().astype(np.uint8), vis_img[..., 3:].numpy().astype(np.uint8)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
    ax1.imshow(img_0)
    ax1.set_title('prev')

    ax2.imshow(img_1)
    ax2.set_title('current')

    ax3.plot(-pred_trajectory[0, :, 1], pred_trajectory[0, :, 0], 'o-', label='pred0 - conf %.3f' % pred_conf[0], alpha=pred_conf[0].clip(0.1))
    ax3.plot(-pred_trajectory[1, :, 1], pred_trajectory[1, :, 0], 'o-', label='pred1 - conf %.3f' % pred_conf[1], alpha=pred_conf[1].clip(0.1))
    ax3.plot(-pred_trajectory[2, :, 1], pred_trajectory[2, :, 0], 'o-', label='pred2 - conf %.3f' % pred_conf[2], alpha=pred_conf[2].clip(0.1))
    ax3.plot(-labels[0, :, 1], labels[0, :, 0], 'o-', label='gt')
    ax3.set_xlim(-30, 30)
    ax3.set_ylim(0, 100)
    ax3.legend()

    ax4.imshow(img_1)
    pred_trajectory = list(pred_trajectory) + [batch['future_poses'].numpy()[0], ]
    pred_conf = list(pred_conf) + [1, ]
    for pred_trajectory_single, pred_conf_single in zip(pred_trajectory, pred_conf):
        location = list((p + camera_translation_inv for p in pred_trajectory_single))
        proj_trajectory = np.array(list((camera_intrinsic @ (camera_rotation_matrix_inv @ l) for l in location)))
        proj_trajectory /= proj_trajectory[..., 2:3].repeat(3, -1)
        proj_trajectory /= 2
        proj_trajectory = proj_trajectory[(proj_trajectory[..., 0] > 0) & (proj_trajectory[..., 0] < 800)]
        proj_trajectory = proj_trajectory[(proj_trajectory[..., 1] > 0) & (proj_trajectory[..., 1] < 450)]
        ax4.plot(proj_trajectory[:, 0], proj_trajectory[:, 1], 'o-', label='pred - conf %.3f' % pred_conf_single, alpha=np.clip(pred_conf_single, 0.1, np.Inf))

    ax4.legend()
    plt.tight_layout()
    plt.savefig('vis_new/%04d.png' % b_idx)
    plt.close(fig)
