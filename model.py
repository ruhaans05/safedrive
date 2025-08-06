# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18

class SafeDriveNet(nn.Module):
    def __init__(self, num_actions=3, seq_len=5):
        super(SafeDriveNet, self).__init__()
        self.seq_len = seq_len
        
        cnn = resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])
        self.cnn_features = 512

        self.rnn = nn.LSTM(
            input_size=self.cnn_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_actions)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.flatten(1)
        x = x.view(B, T, -1)
        x, _ = self.rnn(x)
        x = self.head(x[:, -1, :])

        steer = self.tanh(x[:, 0:1])
        throttle_brake = torch.sigmoid(x[:, 1:])
        return torch.cat([steer, throttle_brake], dim=1)