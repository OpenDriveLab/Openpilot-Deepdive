import os
from re import M
from pytorch_lightning import callbacks
import torch
from torch import nn
import torch.nn.functional as F

from data import PlanningDataset
from model import PlaningNetwork, MultipleTrajectoryPredictionLoss
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import LearningRateMonitor


# Hyper-Parameters
LR = 1e-4
BATCH_SIZE = 32
N_WORKERS = 8

class PlanningBaselineV0(pl.LightningModule):
    def __init__(self, M, num_pts, mtp_alpha) -> None:
        super().__init__()
        self.M = M
        self.num_pts = num_pts
        self.mtp_alpha = mtp_alpha

        self.net = PlaningNetwork(M, num_pts)
        self.mtp_loss = MultipleTrajectoryPredictionLoss(mtp_alpha, M, num_pts, distance_type='angle')

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        inputs, labels = batch['input_img'], batch['future_poses']
        pred_cls, pred_trajectory = self.net(inputs)
        cls_loss, reg_loss = self.mtp_loss(pred_cls, pred_trajectory, labels)
        self.log('loss/cls', cls_loss)
        self.log('loss/reg', reg_loss.mean())
        self.log('loss/reg_x', reg_loss[0])
        self.log('loss/reg_y', reg_loss[1])
        self.log('loss/reg_z', reg_loss[2])

        if batch_idx % 10 == 0:
            pred_trajectory = pred_trajectory.detach().cpu().numpy().reshape(-1, self.M, self.num_pts, 3)
            pred_cls = pred_cls.detach().cpu().numpy()
            fig, ax = plt.subplots()
            ax.plot(-pred_trajectory[0, 0, :, 1], pred_trajectory[0, 0, :, 0], 'o-', label='pred0 - conf %.3f' % pred_cls[0, 0])
            ax.plot(-pred_trajectory[0, 1, :, 1], pred_trajectory[0, 1, :, 0], 'o-', label='pred1 - conf %.3f' % pred_cls[0, 1])
            ax.plot(-pred_trajectory[0, 2, :, 1], pred_trajectory[0, 2, :, 0], 'o-', label='pred2 - conf %.3f' % pred_cls[0, 2])
            ax.plot(-labels.detach().cpu().numpy()[0, :, 1], labels.detach().cpu().numpy()[0, :, 0], 'o-', label='gt')
            plt.legend()
            plt.tight_layout()
            self.logger.experiment.add_figure('test', plt.gcf(), self.global_step)

        return cls_loss + self.mtp_alpha * reg_loss.mean()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LR, weight_decay=0.01)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.9)
        # optimizer = optim.SGD(self.parameters(), lr=LR, momentum=0.9, weight_decay=0.01)
        return [optimizer], [lr_scheduler]


if __name__ == "__main__":

    train = PlanningDataset(split='train')
    val = PlanningDataset(split='val')
    train_loader = DataLoader(train, BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
    val_loader = DataLoader(val, BATCH_SIZE, num_workers=N_WORKERS)

    planning_v0 = PlanningBaselineV0(M=3, num_pts=20, mtp_alpha=1.0)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(gpus=4,
                         accelerator='ddp',
                         profiler='simple',
                         benchmark=True,
                        #  gradient_clip_val=1,
                         log_every_n_steps=1,
                         flush_logs_every_n_steps=20,
                         callbacks=[lr_monitor],
                         )

    trainer.fit(planning_v0, train_loader, val_loader)
