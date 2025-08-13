import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from model import SafeDriveFusionNet # <<< MODIFIED >>>
from dataset import SafeDriveDataset
from config import *
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    full_dataset = SafeDriveDataset(root='carla_data')
    train_size = int(0.8 * len(full_dataset)); val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Model Setup ---
    model = SafeDriveFusionNet(num_actions=3, seq_len=SEQ_LEN).to(device) # <<< MODIFIED >>>
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_val_loss = np.inf
    print(f"\n🚀 Starting training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        model.train()
        # <<< MODIFIED: Unpack lidar from the data loader >>>
        for images, lidars, speeds, actions in train_loader:
            images, lidars, speeds, actions = images.to(device), lidars.to(device), speeds.to(device), actions.to(device)
            
            with autocast(enabled=(device.type == 'cuda')):
                # <<< MODIFIED: Pass all inputs to the model >>>
                outputs = model(images, lidars, speeds)
                loss = criterion(outputs, actions)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, lidars, speeds, actions in val_loader:
                images, lidars, speeds, actions = images.to(device), lidars.to(device), speeds.to(device), actions.to(device)
                outputs = model(images, lidars, speeds)
                val_loss += criterion(outputs, actions).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"✅ Epoch [{epoch+1}/{EPOCHS}] | Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH.replace(".pth", "_best.pth"))
            print(f"🎉 New best model saved.")
        
        scheduler.step()

    print("\n🏁 Final testing...")
    # ... (Final testing loop would also need to be updated similarly) ...

if __name__ == "__main__":
    main()
