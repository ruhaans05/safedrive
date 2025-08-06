# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter  # ← NEW

from model import SafeDriveNet
from dataset import SafeDriveDataset
from config import *

import os
from datetime import datetime
import psutil

def get_memory_usage(): # live memory usage in MB
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # MB

def main():
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Create TensorBoard writer ---
    log_dir = f"runs/safedrive_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📌 TensorBoard logging to: {log_dir}")

    # --- Dataset & DataLoader ---
    dataset = SafeDriveDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == 'cuda'
    )

    # --- Model, Optimizer, Loss ---
    model = SafeDriveNet(num_actions=3, seq_len=SEQ_LEN).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    # --- Mixed Precision ---
    scaler = GradScaler() if device.type == 'cuda' else None

    # --- Training Loop ---
    model.train()
    global_step = 0
    print(f"\n🚀 Starting training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_idx, (images, actions) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            optimizer.zero_grad()

            # --- Forward + Backward (Mixed Precision) ---
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, actions)
                scaler.scale(loss).backward()

                # --- Gradient Clipping ---
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, actions)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
                optimizer.step()

            epoch_loss += loss.item()
            global_step += 1


        
        
            #  --- Log batches and memory usage ---
            if batch_idx % LOG_INTERVAL == 0:
                current_lr = optimizer.param_groups[0]['lr']
                mem_mb = get_memory_usage()
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}, RAM: {mem_mb:.1f} MB")

                # Log to TensorBoard
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("LR", current_lr, global_step)
                writer.add_scalar("Memory/RAM_MB", mem_mb, global_step)

                

        # --- End of epoch ---
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch-level metrics
        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        writer.add_scalar("LR/epoch", current_lr, epoch)

        print(f"✅ Epoch [{epoch+1}/{EPOCHS}] | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    # --- Final save ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Model saved to {MODEL_SAVE_PATH}")
    print(f"📊 TensorBoard logs saved to: {log_dir}")

    # Close writer
    writer.close()

if __name__ == "__main__":
    main()