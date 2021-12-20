import os
import time
import random
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

if torch.__version__ == 'parrots':
    from pavi import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter

from data import PlanningDataset, SequencePlanningDataset, Comma2k19SequenceDataset
from model import PlaningNetwork, MultipleTrajectoryPredictionLoss, SequencePlanningNetwork
from utils import draw_trajectory_on_ax, get_val_metric, get_val_metric_keys


def get_hyperparameters(parser: ArgumentParser):
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)

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

    return parser


def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method='tcp://localhost:%s' % os.environ['PORT'], rank=rank, world_size=world_size)
    print('[%.2f]' % time.time(), 'DDP Initialized at %s:%s' % ('localhost', os.environ['PORT']), rank, 'of', world_size, flush=True)


def get_dataloader(rank, world_size, batch_size, pin_memory=False, num_workers=0):
    train = Comma2k19SequenceDataset('data/comma2k19_train_non_overlap.txt', 's3://comma2k19/','train', use_memcache=True)
    val = Comma2k19SequenceDataset('data/comma2k19_val_debug.txt', 's3://comma2k19/','demo', use_memcache=True)

    if torch.__version__ == 'parrots':
        dist_sampler_params = dict(num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler_params = dict(num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_sampler = DistributedSampler(train, **dist_sampler_params)
    val_sampler = DistributedSampler(val, **dist_sampler_params)

    loader_args = dict(num_workers=num_workers, persistent_workers=True if num_workers > 0 else False, prefetch_factor=2, pin_memory=pin_memory)
    train_loader = DataLoader(train, batch_size, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val, batch_size=1, sampler=val_sampler, **loader_args)

    return train_loader, val_loader


def cleanup():
    dist.destroy_process_group()

class SequenceBaselineV1(nn.Module):
    def __init__(self, M, num_pts, mtp_alpha, lr, optimizer) -> None:
        super().__init__()
        self.M = M
        self.num_pts = num_pts
        self.mtp_alpha = mtp_alpha
        self.lr = lr
        self.optimizer = optimizer

        self.net = SequencePlanningNetwork(M, num_pts)

        self.optimize_per_n_step = 40

    @staticmethod
    def configure_optimizers(args, model):
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
        elif args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, )
        else:
            raise NotImplementedError
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.9)

        return optimizer, lr_scheduler

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros((2, x.size(0), 512)).to(self.device)
        return self.net(x, hidden)


def main(rank, world_size, args):
    if rank == 0:
        writer = SummaryWriter()

    train_dataloader, val_dataloader = get_dataloader(rank, world_size, args.batch_size, False, args.n_workers)
    model = SequenceBaselineV1(args.M, args.num_pts, args.mtp_alpha, args.lr, args.optimizer)
    use_sync_bn = True  # TODO
    if use_sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    optimizer, lr_scheduler = model.configure_optimizers(args, model)
    model: SequenceBaselineV1
    if args.resume and rank == 0:
        print('Loading weights from', args.resume)
        model.load_state_dict(torch.load(args.resume), strict=True)
    dist.barrier()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    loss = MultipleTrajectoryPredictionLoss(args.mtp_alpha, args.M, args.num_pts, distance_type='angle')

    num_steps = 0
    disable_tqdm = (rank != 0)

    for epoch in tqdm(range(args.epochs), disable=disable_tqdm):
        train_dataloader.sampler.set_epoch(epoch)
        
        # for batch_idx, data in enumerate(tqdm(train_dataloader, leave=False, disable=disable_tqdm)):
        #     seq_inputs, seq_labels = data['seq_input_img'].cuda(), data['seq_future_poses'].cuda()
        #     bs = seq_labels.size(0)
        #     seq_length = seq_labels.size(1)
            
        #     hidden = torch.zeros((2, bs, 512)).cuda()
        #     total_loss = 0
        #     for t in tqdm(range(seq_length), leave=False, disable=disable_tqdm):
        #         num_steps += 1
        #         inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
        #         pred_cls, pred_trajectory, hidden = model(inputs, hidden)

        #         cls_loss, reg_loss = loss(pred_cls, pred_trajectory, labels)
        #         total_loss += (cls_loss + args.mtp_alpha * reg_loss.mean()) / model.module.optimize_per_n_step
            
        #         if rank == 0:
        #             # TODO: add a customized log function
        #             writer.add_scalar('loss/cls', cls_loss, num_steps)
        #             writer.add_scalar('loss/reg', reg_loss.mean(), num_steps)
        #             writer.add_scalar('loss/reg_x', reg_loss[0], num_steps)
        #             writer.add_scalar('loss/reg_y', reg_loss[1], num_steps)
        #             writer.add_scalar('loss/reg_z', reg_loss[2], num_steps)
        #             writer.add_scalar('param/lr', optimizer.param_groups[0]['lr'], num_steps)

        #         if (t + 1) % model.module.optimize_per_n_step == 0:
        #             hidden = hidden.clone().detach()
        #             optimizer.zero_grad()
        #             total_loss.backward()
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # TODO: move to args
        #             optimizer.step()
        #             if rank == 0:
        #                 writer.add_scalar('loss/total', total_loss, num_steps)
        #             total_loss = 0

        #     if not isinstance(total_loss, int):
        #         optimizer.zero_grad()
        #         total_loss.backward()
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # TODO: move to args
        #         optimizer.step()
        #         if rank == 0:
        #             writer.add_scalar('loss/total', total_loss, num_steps)

        lr_scheduler.step()
        if (epoch + 1) % 1 == 0:  # TODO: Add to args
            if rank == 0:
                # save model
                ckpt_path = os.path.join(writer.log_dir, 'epoch_%d.pth' % epoch)
                torch.save(model.module.state_dict(), ckpt_path)
                print('[Epoch %d] checkpoint saved at %s' % (epoch, ckpt_path))

            model.eval()
            with torch.no_grad():
                saved_metric_epoch = get_val_metric_keys()
                for batch_idx, data in enumerate(val_dataloader):
                    seq_inputs, seq_labels = data['seq_input_img'].cuda(), data['seq_future_poses'].cuda()

                    bs = seq_labels.size(0)
                    seq_length = seq_labels.size(1)
                    
                    hidden = torch.zeros((2, bs, 512), device=seq_inputs.device)
                    for t in tqdm(range(seq_length), leave=False, disable=disable_tqdm):
                        inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
                        pred_cls, pred_trajectory, hidden = model(inputs, hidden)

                        metrics = get_val_metric(pred_cls, pred_trajectory.view(-1, args.M, args.num_pts, 3), labels)
                        
                        for k, v in metrics.items():
                            saved_metric_epoch[k].append(v.float().mean().item())
                
                dist.barrier()  # Wait for all processes
                # sync
                metric_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.float32, device='cuda')
                counter_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.int32, device='cuda')
                # From Python 3.6 onwards, the standard dict type maintains insertion order by default.
                # But, programmers should not rely on it.
                for i, k in enumerate(sorted(saved_metric_epoch.keys())):
                    metric_single[i] = np.mean(saved_metric_epoch[k])
                    counter_single[i] = len(saved_metric_epoch[k])

                metric_gather = [torch.zeros((len(saved_metric_epoch), ), dtype=torch.float32, device='cuda') for _ in range(world_size)]
                counter_gather = [torch.zeros((len(saved_metric_epoch), ), dtype=torch.int32, device='cuda') for _ in range(world_size)]
                dist.all_gather(metric_gather, metric_single)
                dist.all_gather(counter_gather, counter_single)

                if rank == 0:
                    metric_gather = torch.vstack(metric_gather)  # [world_size, num_metric_keys]
                    counter_gather = torch.vstack(counter_gather)  # [world_size, num_metric_keys]
                    print(metric_gather)
                    print(counter_gather)
                    metric_gather = metric_gather.mean(dim=0)
                    print(metric_gather)
                    for i, k in enumerate(sorted(saved_metric_epoch.keys())):
                        writer.add_scalar(k, metric_gather[i], num_steps)
                dist.barrier()

            model.train()

    cleanup()


if __name__ == "__main__":
    print('[%.2f]' % time.time(), 'starting job...', os.environ['SLURM_PROCID'], 'of', os.environ['SLURM_NTASKS'], flush=True)

    parser = ArgumentParser()
    parser = get_hyperparameters(parser)
    args = parser.parse_args()

    setup(rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['SLURM_NTASKS']))
    main(rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['SLURM_NTASKS']), args=args)
