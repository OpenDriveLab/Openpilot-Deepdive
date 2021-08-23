import os
import json
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PlanningDataset(Dataset):
    def __init__(self, root='data', json_path_pattern='p3_%s.json', split='train'):
        self.samples = json.load(open(os.path.join(root, json_path_pattern % split)))
        print('PlanningDataset: %d samples loaded from %s' % 
              (len(self.samples), os.path.join(root, json_path_pattern % split)))

        self.img_root = os.path.join(root, 'nuscenes')
        self.transforms = transforms.Compose(
            [
                transforms.Resize((900 // 2, 1600 // 2)),
                transforms.ToTensor(),
                transforms.Normalize([0.3890, 0.3937, 0.3851],
                                     [0.2172, 0.2141, 0.2209]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs, future_poses = sample['imgs'], sample['future_poses']

        imgs = list(Image.open(os.path.join(self.img_root, p)) for p in imgs)
        imgs = list(self.transforms(img) for img in imgs)
        input_img = torch.cat(imgs, dim=0)
        
        # process future_poses
        future_poses = torch.tensor(future_poses)
        future_poses[:, 0] = future_poses[:, 0].clamp(1e-2, )  # the car will never go backward

        return input_img, future_poses


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataset = PlanningDataset(split='val')

    calculate_image_mean_std = False
    if calculate_image_mean_std:
        # Calculating the stats of the images
        psum    = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])

        # loop through images
        for img, _ in tqdm(dataset):
            inputs = img[0:3]
            inputs = inputs[None]
            psum    += inputs.sum(axis        = [0, 2, 3])
            psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])
        # pixel count
        count = len(dataset) * 900 * 1600

        # mean and std
        total_mean = psum / count
        total_var  = (psum_sq / count) - (total_mean ** 2)
        total_std  = torch.sqrt(total_var)

        # output
        print('mean: '  + str(total_mean))  # [0.3890, 0.3937, 0.3851]
        print('std:  '  + str(total_std))   # [0.2172, 0.2141, 0.2209]

    calculate_poses_mean_std = True
    if calculate_poses_mean_std:
        import matplotlib.pyplot as plt

        poses_list = []
        for _, poses in tqdm(DataLoader(dataset, shuffle=True)):
            poses_list += list(p.numpy()[None, ] for p in poses[0])
            if len(poses_list) > 40000:
                break

        poses = np.concatenate(poses_list, axis=0)
        print('mean: ', np.mean(poses, axis=0))
        print('std:  ', np.std(poses, axis=0))

        plt.hist(poses[:, 0], bins=200, )
        plt.show()

        plt.hist(poses[:, 1], bins=200, )
        plt.show()

        plt.hist(poses[:, 2], bins=200, )
        plt.show()

    exit()

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for img, poses in dataloader:
        print(img.shape, img.max(), img.min(), img.mean())
        print(poses.shape, poses.max(), poses.min(), poses.mean())
