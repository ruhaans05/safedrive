import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from model import SafeDriveGRU # <<< MODIFIED >>>
from dataset import SafeDriveDataset
from config import *

import os
from datetime import datetime
import psutil
import numpy as np

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_dir = f"runs/safedrive_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📌 TensorBoard logging to: {log_dir}")

    full_dataset = SafeDriveDataset(root='carla_data') # Assuming you have a 'carla_data' folder
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    print(f"Data split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = SafeDriveGRU(num_actions=3, seq_len=SEQ_LEN).to(device) # <<< MODIFIED >>>
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()
    scaler = GradScaler() if device.type == 'cuda' else None

    global_step = 0
    best_val_loss = np.inf
    print(f"\n🚀 Starting training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        model.train()
        train_epoch_loss = 0.0
        # <<< MODIFIED: Unpack speed from the data loader >>>
        for batch_idx, (images, speeds, actions) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            speeds = speeds.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast(enabled=(scaler is not None)):
                # <<< MODIFIED: Pass speeds to the model >>>
                outputs = model(images, speeds)
                loss = criterion(outputs, actions)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
                optimizer.step()

            train_epoch_loss += loss.item()
            global_step += 1
        
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            # <<< MODIFIED: Unpack speed for validation >>>
            for images, speeds, actions in val_loader:
                images = images.to(device, non_blocking=True)
                speeds = speeds.to(device, non_blocking=True)
                actions = actions.to(device, non_blocking=True)
                # <<< MODIFIED: Pass speeds to the model >>>
                outputs = model(images, speeds)
                loss = criterion(outputs, actions)
                val_epoch_loss += loss.item()

        avg_train_loss = train_epoch_loss / len(train_loader)
        avg_val_loss = val_epoch_loss / len(val_loader)
        print(f"✅ Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = MODEL_SAVE_PATH.replace(".pth", "_best.pth") 
            torch.save(model.state_dict(), best_model_path)
            print(f"🎉 New best model saved with validation loss: {best_val_loss:.4f}")
        
        scheduler.step()

    print("\n🏁 Starting final testing...")
    best_model_path = MODEL_SAVE_PATH.replace(".pth", "_best.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, speeds, actions in test_loader:
            images = images.to(device, non_blocking=True)
            speeds = speeds.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            outputs = model(images, speeds)
            loss = criterion(outputs, actions)
            test_loss += loss.item()
            
    avg_test_loss = test_loss / len(test_loader)
    print(f"🏆 Final Test Loss: {avg_test_loss:.4f}")

    writer.close()

if __name__ == "__main__":
    main()
