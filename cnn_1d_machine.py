import torch
import torch.nn as nn

class Machine(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_net = nn.Sequential(
            nn.Conv1d(1,8,5),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(8, 16, 5),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, 5),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Flatten(),
            nn.Linear(64,1)
        )

    def forward(self, x):
        x = self.band_net(x)
        return x