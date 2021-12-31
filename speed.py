import torch
from tqdm import tqdm

from main import SequenceBaselineV1


planning_v0 = SequenceBaselineV1(5, 33, 1.0, 0.0, 'adamw')
planning_v0.eval().cuda()

dummy_input = torch.rand((1, 6, 128, 256), device='cuda')
hidden = torch.rand((2, 1, 512), device='cuda')

with torch.no_grad():
    for b_idx in tqdm(range(1000)):
        _, _, hidden = planning_v0(dummy_input, hidden)
