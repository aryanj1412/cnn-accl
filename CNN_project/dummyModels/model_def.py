# dummy_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.fc1 = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)   # 32x32 → 16x16

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)   # 16x16 → 8x8

        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x