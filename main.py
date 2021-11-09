import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from argparse import ArgumentParser

from data import PlanningDataset, SequencePlanningDataset
from model import PlaningNetwork, MultipleTrajectoryPredictionLoss, SequencePlanningNetwork
from utils import draw_trajectory_on_ax, get_val_metric
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import LearningRateMonitor


class PlanningBaselineV0(pl.LightningModule):
    def __init__(self, M, num_pts, mtp_alpha, lr) -> None:
        super().__init__()
        self.M = M
        self.num_pts = num_pts
        self.mtp_alpha = mtp_alpha
        self.lr = lr

        self.net = PlaningNetwork(M, num_pts)
        self.mtp_loss = MultipleTrajectoryPredictionLoss(mtp_alpha, M, num_pts, distance_type='angle')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('PlanningBaselineV0')
        parser.add_argument('--M', type=int, default=3)
        parser.add_argument('--num_pts', type=int, default=20)
        parser.add_argument('--mtp_alpha', type=float, default=1.0)
        return parent_parser

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
            trajectories = list(pred_trajectory[0].detach().cpu().numpy().reshape(self.M, self.num_pts, 3))  # M, num_pts, 3
            trajectories.append(labels[0].detach().cpu().numpy())
            confs = list(F.softmax(pred_cls[0].detach().cpu(), dim=-1).numpy()) + [1, ] # M,

            fig, ax = plt.subplots()
            ax = draw_trajectory_on_ax(ax, trajectories, confs)
            plt.tight_layout()
            self.logger.experiment.add_figure('train_vis', fig, self.global_step)
            plt.close(fig)

        return cls_loss + self.mtp_alpha * reg_loss.mean()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.01)
        # optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.9)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['input_img'], batch['future_poses']
        pred_cls, pred_trajectory = self.net(inputs)

        metrics = get_val_metric(pred_cls, pred_trajectory.view(-1, self.M, self.num_pts, 3), labels)

        self.log_dict(metrics)
        # Pytorch-lightning will collect those binary values and calculate the mean


class SequencePlanningBaselineV0(pl.LightningModule):
    def __init__(self, M, num_pts, mtp_alpha, lr, optimizer) -> None:
        super().__init__()
        self.M = M
        self.num_pts = num_pts
        self.mtp_alpha = mtp_alpha
        self.lr = lr
        self.optimizer = optimizer

        self.net = SequencePlanningNetwork(M, num_pts)
        self.mtp_loss = MultipleTrajectoryPredictionLoss(mtp_alpha, M, num_pts, distance_type='angle')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('SequencePlanningBaselineV0')
        parser.add_argument('--M', type=int, default=3)
        parser.add_argument('--num_pts', type=int, default=20)
        parser.add_argument('--mtp_alpha', type=float, default=1.0)
        parser.add_argument('--optimizer', type=str, default='sgd')
        return parent_parser

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros((2, x.size(0), 512)).to(self.device)
        # in lightning, forward defines the prediction/inference actions
        return self.net(x, hidden)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        seq_inputs, seq_labels = batch['seq_input_img'], batch['seq_future_poses']
        bs = seq_labels.size(0)
        seq_length = seq_labels.size(1)
        
        cls_loss_total, reg_loss_total = 0, 0
        hidden = torch.zeros((2, bs, 512)).to(self.device)
        for t in range(seq_length):
            inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
            pred_cls, pred_trajectory, hidden = self.net(inputs, hidden)
            cls_loss, reg_loss = self.mtp_loss(pred_cls, pred_trajectory, labels)
            cls_loss_total += cls_loss
            reg_loss_total += reg_loss

        cls_loss_total = cls_loss_total / (seq_length + 1)
        reg_loss_total = reg_loss_total / (seq_length + 1)
        
        self.log('loss/cls', cls_loss_total)
        self.log('loss/reg', reg_loss_total.mean())
        self.log('loss/reg_x', reg_loss_total[0])
        self.log('loss/reg_y', reg_loss_total[1])
        self.log('loss/reg_z', reg_loss_total[2])

        # if batch_idx % 10 == 0:
        #     trajectories = list(pred_trajectory[0].detach().cpu().numpy().reshape(self.M, self.num_pts, 3))  # M, num_pts, 3
        #     trajectories.append(labels[0].detach().cpu().numpy())
        #     confs = list(F.softmax(pred_cls[0].detach().cpu(), dim=-1).numpy()) + [1, ] # M,

        #     fig, ax = plt.subplots()
        #     ax = draw_trajectory_on_ax(ax, trajectories, confs)
        #     plt.tight_layout()
        #     self.logger.experiment.add_figure('train_vis', fig, self.global_step)
        #     plt.close(fig)

        return cls_loss + self.mtp_alpha * reg_loss.mean()

    def configure_optimizers(self):
        if self.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.01)
        elif self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        else:
            raise NotImplementedError
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.9)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        seq_inputs, seq_labels = batch['seq_input_img'], batch['seq_future_poses']

        bs = seq_labels.size(0)
        seq_length = seq_labels.size(1)
        
        hidden = torch.zeros((2, bs, 512)).to(self.device)
        for t in range(seq_length):
            inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
            pred_cls, pred_trajectory, hidden = self.net(inputs, hidden)

            metrics = get_val_metric(pred_cls, pred_trajectory.view(-1, self.M, self.num_pts, 3), labels)
            self.log_dict(metrics)


if __name__ == "__main__":

    parser = ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=32 * 4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_workers', type=int, default=8)

    parser = SequencePlanningBaselineV0.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()

    data_p = {10: 'p3_10pts_%s.json', 20: 'p3_%s.json'}[args.num_pts]
    data_p = 'p3_10pts_can_bus_%s_temporal.json'
    train = SequencePlanningDataset(split='train', json_path_pattern=data_p)
    val = SequencePlanningDataset(split='val', json_path_pattern=data_p)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=args.n_workers, persistent_workers=True, prefetch_factor=4, pin_memory=True)
    val_loader = DataLoader(val, args.batch_size, num_workers=args.n_workers, persistent_workers=True, prefetch_factor=4, pin_memory=True)

    planning_v0 = SequencePlanningBaselineV0(args.M, args.num_pts, args.mtp_alpha, args.lr, args.optimizer)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer.from_argparse_args(args,
                                            accelerator='ddp' if args.gpus > 1 else None,
                                            profiler='simple',
                                            benchmark=True,
                                            log_every_n_steps=10,
                                            flush_logs_every_n_steps=50,
                                            callbacks=[lr_monitor],
                                            check_val_every_n_epoch=10,  # val every 10 epoch to speed up train process
                                            # val_check_interval=0.0,  # Disable in-batch val
                                            progress_bar_refresh_rate=10,  # for slurm env
                                            )

    trainer.fit(planning_v0, train_loader, val_loader)
