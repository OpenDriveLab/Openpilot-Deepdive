from utils import draw_trajectory_on_ax
import torch
import numpy as np
from torch.nn.functional import softmax

from data import PlanningDataset, SequencePlanningDataset
from main import PlanningBaselineV0, SequencePlanningBaselineV0
from torch.utils.data import DataLoader
import pytorch_lightning as pl



data_p = 'p3_10pts_can_bus_%s_temporal.json'
val = SequencePlanningDataset(split='val', json_path_pattern=data_p)
val_loader = DataLoader(val, 1, num_workers=1, persistent_workers=True, prefetch_factor=4, pin_memory=True)

planning_v0 = SequencePlanningBaselineV0.load_from_checkpoint('epoch=999-step=154999.ckpt', M=3, num_pts=10, mtp_alpha=1.0, lr=0)

trainer = pl.Trainer(gpus=1)

trainer.validate(planning_v0, val_loader, 'epoch=999-step=154999.ckpt')
