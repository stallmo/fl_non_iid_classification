import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetCifar10(nn.Module):
    def __init__(self):
        super().__init__()
        # This is the "LeNet" like architecture from the tensorflow tutorial as referenced in the papers about non iid federated learning
        self.conv1 = nn.Conv2d(3, 32, 3) # input are 3 channels (32x32 RGB images). Output are 32 channels
        self.pool = nn.MaxPool2d(2, 2) # 2x2 pooling will output half the size of the input
        self.conv2 = nn.Conv2d(32, 64, 3) # input are 32 channels, output are 64 channels
        self.conv3 = nn.Conv2d(64, 64,3) # input are 64 channels, output are 64 channels
        self.fc1 = nn.Linear(64 * 4 * 4, 64) # Input is 32*32, 2 pooling and 3 convolution layers reduce it to 4*4, 64 is the number of output channels of the last conv layer
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x