import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from argparse import ArgumentParser
from PIL import Image

from data import PlanningDataset
from model import PlaningNetwork
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from efficientnet_pytorch import EfficientNet


class JudgeDataset(PlanningDataset):
    def __init__(self, json_path_pattern, split):
        super().__init__(json_path_pattern=json_path_pattern, split=split)

        self.intrinsic_dict = {1252: 0, 1262: 1, 1266: 2}

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs = sample['imgs']

        img = self.transforms(Image.open(os.path.join(self.img_root, imgs[0])))
        intrinsic_type=int(sample['camera_intrinsic'][0][0])

        return dict(
            input_img=img,
            cls=self.intrinsic_dict[intrinsic_type]
        )


class PlanningBaselineV0(pl.LightningModule):
    def __init__(self, lr) -> None:
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b2', num_classes=3)
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        inputs, labels = batch['input_img'], batch['cls']
        pred_cls = self.net(inputs)
        cls_loss = self.loss(pred_cls, labels)
        self.log('loss/cls', cls_loss)

        return cls_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.01)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.9)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['input_img'], batch['cls']
        pred_cls = self.net(inputs)

        metrics = {'val_acc': torch.argmax(pred_cls, -1) == labels}

        self.log_dict(metrics)
        # Pytorch-lightning will collect those binary values and calculate the mean


if __name__ == "__main__":

    parser = ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_workers', type=int, default=8)

    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()

    train = JudgeDataset(split='train', json_path_pattern='p3_10pts_%s.json')
    val = JudgeDataset(split='val', json_path_pattern='p3_10pts_%s.json')
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=args.n_workers)
    val_loader = DataLoader(val, args.batch_size, num_workers=args.n_workers)

    planning_v0 = PlanningBaselineV0(args.lr)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer.from_argparse_args(args,
                                            accelerator='ddp',
                                            profiler='simple',
                                            benchmark=True,
                                            log_every_n_steps=10,
                                            flush_logs_every_n_steps=50,
                                            callbacks=[lr_monitor],
                                            )

    trainer.fit(planning_v0, train_loader, val_loader)
