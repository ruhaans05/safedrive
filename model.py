# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18

class SafeDriveNet(nn.Module):
    def __init__(self, num_actions=3, seq_len=5):
        super(SafeDriveNet, self).__init__()
        self.seq_len = seq_len
        
        # CNN backbone (ResNet18)
        cnn = resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])  # Remove fc
        self.cnn_features = 512

        # RNN
        self.rnn = nn.LSTM(
            input_size=self.cnn_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Final head
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_actions)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # Merge batch and time

        x = self.cnn(x)             # (B*T, 512, 1, 1)
        x = x.flatten(1)            # (B*T, 512)
        x = x.view(B, T, -1)        # (B, T, 512)

        x, _ = self.rnn(x)          # (B, T, 128)
        x = self.head(x[:, -1, :])  # Last timestep

        # steer ∈ [-1, 1], throttle/brake ∈ [0, 1]
        steer = self.tanh(x[:, 0:1])
        throttle_brake = torch.sigmoid(x[:, 1:])
        return torch.cat([steer, throttle_brake], dim=1)
