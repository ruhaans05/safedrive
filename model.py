import torch
import torch.nn as nn
from torchvision.models import resnet18

class SafeDriveGRU(nn.Module): # <<< RENAMED and MODIFIED >>>
    def __init__(self, num_actions=3, seq_len=5):
        super(SafeDriveGRU, self).__init__()
        self.seq_len = seq_len
        
        # CNN Feature Extractor
        cnn = resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])
        self.cnn_features = 512

        # GRU for temporal processing
        self.rnn = nn.GRU( # <<< CHANGED from LSTM to GRU >>>
            input_size=self.cnn_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Decision-making head
        self.head = nn.Sequential(
            # <<< MODIFIED: Input size is now GRU output (128) + speed (1) >>>
            nn.Linear(128 + 1, 64), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_actions)
        )
        self.tanh = nn.Tanh()

    def forward(self, x_img, x_speed): # <<< MODIFIED: Accepts image and speed >>>
        B, T, C, H, W = x_img.shape
        x_img = x_img.view(B * T, C, H, W)
        
        # Process images through CNN
        x_img = self.cnn(x_img)
        x_img = x_img.flatten(1)
        x_img = x_img.view(B, T, -1)
        
        # Process sequence through GRU
        x_img, _ = self.rnn(x_img)
        
        # Get the output from the last time step
        rnn_out = x_img[:, -1, :]
        
        # <<< NEW: Concatenate GRU output with speed measurement >>>
        combined_features = torch.cat([rnn_out, x_speed], dim=1)
        
        # Pass combined features to the head
        x = self.head(combined_features)

        # Apply activation functions to actions
        steer = self.tanh(x[:, 0:1])
        throttle_brake = torch.sigmoid(x[:, 1:])
        return torch.cat([steer, throttle_brake], dim=1)
