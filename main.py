import os
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from data import PlanningDataset, SequencePlanningDataset, Comma2k19SequenceDataset
from model import PlaningNetwork, MultipleTrajectoryPredictionLoss, SequencePlanningNetwork
from utils import draw_trajectory_on_ax, get_val_metric


def get_hyperparameters(parser: ArgumentParser):
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--resume', type=str, default='')

    parser.add_argument('--M', type=int, default=3)
    parser.add_argument('--num_pts', type=int, default=20)
    parser.add_argument('--mtp_alpha', type=float, default=1.0)
    parser.add_argument('--optimizer', type=str, default='sgd')

    try:
        exp_name = os.environ["SLURM_JOB_ID"]
    except KeyError:
        exp_name = str(time.time())
    parser.add_argument('--exp_name', type=str, default=exp_name)


def setup(rank, world_size):
    master_addr = 'localhost'
    master_port = random.randint(30000, 50000)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print('Distributed Environment Initialized at %s:%s' % (master_addr, master_port))


def get_dataloader(rank, world_size, batch_size, pin_memory=False, num_workers=0):
    train = Comma2k19SequenceDataset('data/comma2k19_val_non_overlap.txt', 'data/comma2k19/','train', use_memcache=False)
    val = Comma2k19SequenceDataset('data/comma2k19_val_non_overlap.txt', 'data/comma2k19/','val', use_memcache=False)

    train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)

    loader_args = dict(num_workers=num_workers, persistent_workers=True, prefetch_factor=2, pin_memory=pin_memory)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val, args.batch_size, sampler=val_sampler, **loader_args)

    return train_loader, val_loader


def cleanup():
    dist.destroy_process_group()

class SequenceBaselineV1(nn.Module):
    def __init__(self, M, num_pts, mtp_alpha, lr, optimizer) -> None:
        super().__init__(M, num_pts, mtp_alpha, lr, optimizer)
        self.automatic_optimization = False
        self.optimize_per_n_step = 40

    def training_step(self, batch, batch_idx=-1):
        # manual backward
        opt = self.optimizers()

        seq_inputs, seq_labels = batch['seq_input_img'], batch['seq_future_poses']
        bs = seq_labels.size(0)
        seq_length = seq_labels.size(1)
        
        hidden = torch.zeros((2, bs, 512)).to(self.device)
        total_loss = 0
        for t in tqdm(range(seq_length), leave=False):
            inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
            pred_cls, pred_trajectory, hidden = self.net(inputs, hidden)
            cls_loss, reg_loss = self.mtp_loss(pred_cls, pred_trajectory, labels)
            total_loss += (cls_loss + self.mtp_alpha * reg_loss.mean()) / self.optimize_per_n_step
        
            # self.log('loss/cls', cls_loss)
            # self.log('loss/reg', reg_loss.mean())
            # self.log('loss/reg_x', reg_loss[0])
            # self.log('loss/reg_y', reg_loss[1])
            # self.log('loss/reg_z', reg_loss[2])
            # TODO

            if (t + 1) % self.optimize_per_n_step == 0:
                opt.zero_grad()
                self.manual_backward(total_loss)
                opt.step()
                hidden = hidden.clone().detach()
                total_loss = 0

        if not isinstance(total_loss, int):
            opt.zero_grad()
            self.manual_backward(total_loss)
            opt.step()

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


def main(rank, world_size, args):
    setup(rank, world_size)

    train_dataloader, val_dataloader = get_dataloader(rank, world_size, args.batch_size, args.n_workers)
    model = SequenceBaselineV1(args.M, args.num_pts, args.mtp_alpha, args.lr, args.optimizer).cuda()
    model: SequenceBaselineV1
    if args.resume:
        model.load_state_dict(args.resume, strict=True)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)

    optimizer, scheduler = model.configure_optimizers()

    for epoch in tqdm(range(args.epochs)):
        for batch_idx, data in enumerate(train_dataloader):
            data = data.cuda()
            model.training_step(data)

        if (epoch + 1) % 10 == 0:
            for batch_idx, data in enumerate(val_dataloader):
                data = data.cuda()
                model.validation_step(data)  # TODO

    cleanup()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = SequencePlanningBaselineV0.add_model_specific_args(parser)
    args = parser.parse_args()

    world_size = args.gpus
    mp.spawn(
        main,
        args=(world_size, args),
        nprocs=world_size
    )
