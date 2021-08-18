import torch
import torch.nn as nn

from tqdm import tqdm
from data import PlanningDataset
from model import PlaningNetwork, MultipleTrajectoryPredictionLoss
from torch import optim
from torch.utils.data import DataLoader


def main(model, train_loader, val_loader, criterion, optimizer, scheduler):
    device = 'cuda:0'
    model.to(device)

    for epoch in tqdm(range(EPOCHS), ncols=0, postfix='Epoch'):
        # Training
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            pred_cls, pred_trajectory = model(inputs)
            cls_loss, reg_loss = criterion(pred_cls, pred_trajectory, labels)

            loss = cls_loss + MTPLOSS_ALPHA * reg_loss
            loss.backward()
            optimizer.step()

            print(loss.item())

        scheduler.step()


if __name__ == "__main__":
    # HyperParameters
    BATCH_SIZE = 4
    LR = 1e-3
    EPOCHS = 20
    MTPLOSS_ALPHA = 1.0

    model = PlaningNetwork(3, 20)
    train_data = PlanningDataset(split='train')
    val_data = PlanningDataset(split='val')
    criterion = MultipleTrajectoryPredictionLoss(MTPLOSS_ALPHA, 3, 20, )

    train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, BATCH_SIZE, shuffle=False)

    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    main(model, train_loader, val_loader, criterion, optimizer, scheduler)
