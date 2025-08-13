# config.py
import torch
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "carla_data")  # will hold real data
IMG_HEIGHT = 224
IMG_WIDTH = 224
SEQ_LEN = 5
BATCH_SIZE = 8 if torch.cuda.is_available() else 4 #8 is good for a 8-16 gb ram GPU (we give 8 is CUDA GPU is available)
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_VAL = 1.0
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "safedrive_model.pth")
LOG_INTERVAL = 10
