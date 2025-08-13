import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import json
from config import *

class SafeDriveDataset(Dataset):
    def __init__(self, root='carla_data', seq_len=SEQ_LEN):
        self.root = root
        self.seq_len = seq_len
        self.image_dir = os.path.join(root, 'front')
        self.lidar_dir = os.path.join(root, 'lidar') # <<< NEW >>>
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
        controls_path = os.path.join(root, 'controls.json')
        with open(controls_path, 'r') as f:
            self.controls = {item['frame']: item for item in json.load(f)}

    def __len__(self):
        return len(self.image_files) - self.seq_len

    def __getitem__(self, idx):
        # Image sequence
        images = []
        for i in range(idx, idx + self.seq_len):
            img_path = os.path.join(self.image_dir, self.image_files[i])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            images.append(img)
        images = torch.from_numpy(np.stack(images, axis=0))

        # <<< NEW: Load LiDAR data for the last frame in the sequence >>>
        lidar_path = os.path.join(self.lidar_dir, self.image_files[idx + self.seq_len - 1])
        lidar_img = cv2.imread(lidar_path, cv2.IMREAD_GRAYSCALE)
        lidar_img = lidar_img.astype(np.float32) / 255.0
        lidar_tensor = torch.from_numpy(lidar_img).unsqueeze(0) # Add channel dimension

        # Controls and speed
        target_frame_id = idx + self.seq_len - 1
        ctrl = self.controls.get(target_frame_id, {'steer': 0.0, 'throttle': 0.0, 'brake': 0.0, 'speed_kmh': 0.0})
        action = torch.tensor([ctrl['steer'], ctrl['throttle'], ctrl['brake']], dtype=torch.float32)
        speed = torch.tensor([ctrl['speed_kmh']], dtype=torch.float32)

        return images, lidar_tensor, speed, action
