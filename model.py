import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Conv2d(i, o, k, p) >> 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))     # (batch, 32, 28, 28)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 14, 14)
        x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7)    # Flaten, make shape(batch, 64 * 7 * 7) for fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)               # make this value to probability by softmax later
        return x