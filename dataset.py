# dataset.py
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from utils import create_dummy_image

class SafeDriveDataset(Dataset):
    def __init__(self, root="dummy_data", seq_len=5, transform=None):
        self.root = root
        self.seq_len = seq_len
        self.transform = transform
        
        # Simulate sequences: dummy_data/seq_00001/, seq_00002/, etc.
        if not os.path.exists(root):
            print(f"[INFO] No data found in {root}, generating dummy dataset...")
            self._generate_dummy_data(n_seqs=100, seq_len=seq_len)
        
        self.sequences = [d for d in os.listdir(root) if d.startswith("seq_")]
        self.sequences.sort()
        
        # Each sequence has `seq_len` or more frames
        self.items = []
        for seq in self.sequences:
            seq_path = os.path.join(root, seq)
            num_frames = len([f for f in os.listdir(seq_path) if f.endswith(".png")])
            if num_frames >= seq_len:
                for end_idx in range(seq_len, num_frames + 1):
                    self.items.append((seq, end_idx))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq_name, end_frame = self.items[idx]
        seq_path = os.path.join(self.root, seq_name)
        
        # Load sequence of images
        images = []
        for i in range(end_frame - self.seq_len, end_frame):
            img_path = os.path.join(seq_path, f"{i:06d}.png")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))  # Match model input
            else:
                img = create_dummy_image()  # fallback
            if self.transform:
                img = self.transform(img)
            else:
                img = img.astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
            images.append(img)
        
        images = torch.stack(images, dim=0)  # (T, C, H, W)

        # Dummy action: [steer, throttle, brake]
        action = np.random.uniform(-1, 1, (3,)).astype(np.float32)
        action[1] = max(action[1], 0)  # throttle ≥ 0
        action[2] = max(action[2], 0)  # brake ≥ 0
        action = torch.from_numpy(action)

        return images, action

    def _generate_dummy_data(self, n_seqs=100, seq_len=5):
        os.makedirs(self.root, exist_ok=True)
        for seq_id in range(n_seqs):
            seq_folder = os.path.join(self.root, f"seq_{seq_id:05d}")
            os.makedirs(seq_folder, exist_ok=True)
            for frame_id in range(seq_len + 5):  # extra frames
                img = create_dummy_image()
                cv2.imwrite(os.path.join(seq_folder, f"{frame_id:06d}.png"), img)
        print(f"[INFO] Generated {n_seqs} dummy sequences in {self.root}")
