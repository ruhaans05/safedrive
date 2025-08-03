# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SafeDriveNet
from dataset import SafeDriveDataset
from utils import DEVICE

def main():
    print(f"Using device: {DEVICE}")

    # Dataset & DataLoader
    dataset = SafeDriveDataset(seq_len=5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # Model
    model = SafeDriveNet(num_actions=3, seq_len=5).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    model.train()
    print("Starting training...\n")

    for epoch in range(5):
        epoch_loss = 0.0
        for batch_idx, (images, actions) in enumerate(dataloader):
            images = images.to(DEVICE)   # (B, T, C, H, W)
            actions = actions.to(DEVICE) # (B, 3)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/5], Loss: {avg_loss:.4f}")

    print("\n✅ Training complete! Framework is ready.")

    # Save model
    torch.save(model.state_dict(), "safedrive_model.pth")
    print("Model saved to safedrive_model.pth")

if __name__ == "__main__":
    main()
