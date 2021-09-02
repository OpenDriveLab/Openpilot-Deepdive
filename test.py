from utils import draw_trajectory_on_ax
import torch
import numpy as np
from torch.nn.functional import softmax

from data import PlanningDataset
from main import PlanningBaselineV0
from torch.utils.data import DataLoader
import pytorch_lightning as pl



val = PlanningDataset(split='val')
val_loader = DataLoader(val, 16, num_workers=4, shuffle=False)

planning_v0 = PlanningBaselineV0.load_from_checkpoint('epoch=910-step=92921.ckpt', M=3, num_pts=20, mtp_alpha=1.0, lr=0)

trainer = pl.Trainer(gpus=1)

trainer.validate(planning_v0, val_loader, 'epoch=910-step=92921.ckpt')
