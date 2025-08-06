# dataset.py
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from config import *

class SafeDriveDataset(Dataset):
    def __init__(self, root='carla_data', seq_len=SEQ_LEN, transform=None):
        self.root = root
        self.seq_len = seq_len
        self.transform = transform

        # Get all front camera images (sorted by name)
        self.image_dir = os.path.join(root, 'front')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])

        # Control files
        self.control_dir = os.path.join(root, 'controls')

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        # Number of sequences
        return len(self.image_files) - self.seq_len + 1

    def __getitem__(self, idx):
        # Load sequence of images (t-4, t-3, t-2, t-1, t)
        images = []
        for i in range(idx, idx + self.seq_len):
            img_path = os.path.join(self.image_dir, self.image_files[i])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW
            images.append(img)

        # Stack into (T, C, H, W)
        images = torch.from_numpy(np.stack(images, axis=0))

        # Load action at time t (last frame in sequence)
        ctrl_path = os.path.join(self.control_dir, f"{idx + self.seq_len - 1:06d}.npy")
        if os.path.exists(ctrl_path):
            ctrl = np.load(ctrl_path, allow_pickle=True).item()
        else:
            # Fallback (should not happen)
            ctrl = {'steer': 0.0, 'throttle': 0.0, 'brake': 0.0}

        action = np.array([ctrl['steer'], ctrl['throttle'], ctrl['brake']], dtype=np.float32)
        action = torch.from_numpy(action)

        return images, action