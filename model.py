import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


class PlaningNetwork(nn.Module):
    def __init__(self, M, num_pts):
        super().__init__()
        self.M = M
        self.num_pts = num_pts
        self.backbone = EfficientNet.from_pretrained('efficientnet-b2', in_channels=6)
        self.plan_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1408, M * num_pts * 3),
        )

    def forward(self, x):
        features = self.backbone.extract_features(x)
        pred = self.plan_head(features)
        return pred.reshape(-1, self.M, self.num_pts * 3)


if __name__ == '__main__':
    # model = EfficientNet.from_pretrained('efficientnet-b2', in_channels=6)
    model = PlaningNetwork(M=3, num_pts=20)

    dummy_input = torch.zeros((1, 6, 256, 512))

    # features = model.extract_features(dummy_input)
    features = model(dummy_input)

    print(features.shape)
