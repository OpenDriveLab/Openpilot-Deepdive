import os
from re import M
import torch
from torch import nn
import torch.nn.functional as F

from data import PlanningDataset
from model import PlaningNetwork, MultipleTrajectoryPredictionLoss
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl


# Hyper-Parameters
LR = 1e-2
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
        inputs, labels = batch
        pred_cls, pred_trajectory = self.net(inputs)
        cls_loss, reg_loss = self.mtp_loss(pred_cls, pred_trajectory, labels)
        self.log('loss/cls', cls_loss)
        self.log('loss/reg', reg_loss.mean())
        self.log('loss/reg_x', reg_loss[0])
        self.log('loss/reg_y', reg_loss[1])
        self.log('loss/reg_z', reg_loss[2])
        return cls_loss + self.mtp_alpha * reg_loss.mean()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LR)
        return optimizer


if __name__ == "__main__":

    train = PlanningDataset(split='train')
    val = PlanningDataset(split='val')
    train_loader = DataLoader(train, BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
    val_loader = DataLoader(val, BATCH_SIZE, num_workers=N_WORKERS)

    planning_v0 = PlanningBaselineV0(M=3, num_pts=20, mtp_alpha=1.0)
    trainer = pl.Trainer(gpus=4,
                         accelerator='ddp',
                         profiler='simple',
                         benchmark=True,
                         gradient_clip_val=1,
                         )

    trainer.fit(planning_v0, train_loader, val_loader)
