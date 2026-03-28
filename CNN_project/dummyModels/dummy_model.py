import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity   # Residual connection
        return F.relu(out)


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.block1 = TinyResidualBlock(8)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 16 * 16, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = TinyResNet()

    # Dummy training (just to initialize weights)
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(5):  # 5 iterations only
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Save FULL model (important!)
    torch.save(model, "model.pt")

    print("Dummy model saved as model.pt")