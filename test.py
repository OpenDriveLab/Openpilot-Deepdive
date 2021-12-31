from utils import draw_trajectory_on_ax
import torch
import numpy as np
from torch.nn.functional import softmax

from data import PlanningDataset, SequencePlanningDataset, Comma2k19SequenceDataset
from main import PlanningBaselineV0, SequencePlanningBaselineV0, SequenceBaselineV1
from torch.utils.data import DataLoader
import pytorch_lightning as pl



val = Comma2k19SequenceDataset('data/comma2k19_val_non_overlap.txt', 'data/comma2k19/','val', use_memcache=False)
val_loader = DataLoader(val, 1, num_workers=1, persistent_workers=True, prefetch_factor=1, pin_memory=True)

planning_v0 = SequenceBaselineV1.load_from_checkpoint('comma2k19-bs=6-epoch=17.ckpt', M=3, num_pts=33, mtp_alpha=1.0, lr=0, optimizer='adam')

trainer = pl.Trainer(gpus=1)

trainer.validate(planning_v0, val_loader, 'comma2k19-bs=6-epoch=17.ckpt')
