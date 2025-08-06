# utils.py
import numpy as np
import cv2
from torchvision import transforms

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_dummy_image():
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)