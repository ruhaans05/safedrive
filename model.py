import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SelfAttention(nn.Module): # <<< NEW: Attention Mechanism >>>
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, T, C) -> (B, C, T)
        proj_query = self.query_conv(x).permute(0, 2, 1) # (B, T, C')
        proj_key = self.key_conv(x) # (B, C', T)
        energy = torch.bmm(proj_query, proj_key) # (B, T, T)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).permute(0, 2, 1) # (B, T, C)
        out = torch.bmm(attention, proj_value)
        out = self.gamma * out + x.permute(0, 2, 1)
        return out, attention

class SafeDriveFusionNet(nn.Module): # <<< NEW: Fusion Model >>>
    def __init__(self, num_actions=3, seq_len=5):
        super(SafeDriveFusionNet, self).__init__()
        
        # --- Vision Branch ---
        self.cnn = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        self.gru = nn.GRU(input_size=512, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        self.attention = SelfAttention(in_dim=128)

        # --- LiDAR Branch ---
        self.lidar_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 64), nn.ReLU()
        )

        # --- Fusion and Decision Head ---
        self.head = nn.Sequential(
            # Input: Attention Out (128) + LiDAR Out (64) + Speed (1)
            nn.Linear(128 + 64 + 1, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        self.tanh = nn.Tanh()

    def forward(self, x_img, x_lidar, x_speed):
        B, T, C, H, W = x_img.shape
        
        # Process Vision
        img_features = self.cnn(x_img.view(B * T, C, H, W)).flatten(1)
        img_features = img_features.view(B, T, -1)
        gru_out, _ = self.gru(img_features)
        attn_out, _ = self.attention(gru_out)
        vision_final_features = attn_out[:, -1, :] # Use last time step's attended features

        # Process LiDAR
        lidar_features = self.lidar_cnn(x_lidar)

        # Fusion
        combined_features = torch.cat([vision_final_features, lidar_features, x_speed], dim=1)
        
        # Decision
        actions = self.head(combined_features)
        steer = self.tanh(actions[:, 0:1])
        throttle_brake = torch.sigmoid(actions[:, 1:])
        return torch.cat([steer, throttle_brake], dim=1)
