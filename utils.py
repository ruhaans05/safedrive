# utils.py
import torch
import numpy as np
from torchvision import transforms

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transform
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

# Generate dummy image (64x64 RGB)
def create_dummy_image():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return img
