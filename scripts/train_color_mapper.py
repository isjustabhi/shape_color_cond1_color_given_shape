import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Add root to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from color_dataset import ColorMappingDataset
from color_mapper import ColorMapper

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 32
SAMPLE_EVERY = 5
EDGE_WEIGHT = 0.1  # Weight for edge-aware loss

# Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("samples/cond1", exist_ok=True)

# Load dataset
dataset = ColorMappingDataset("data/toy_dataset/train")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = ColorMapper().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Edge-aware loss
def edge_loss(pred, target):
    # Crop pred and target to same shape before slicing
    H = min(pred.shape[2], target.shape[2])
    W = min(pred.shape[3], target.shape[3])
    pred = pred[:, :, :H, :W]
    target = target[:, :, :H, :W]

    # Compute gradients
    pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

    # Crop gradients to match each other
    grad_H = min(pred_grad_y.shape[2], target_grad_y.shape[2])
    grad_W = min(pred_grad_x.shape[3], target_grad_x.shape[3])

    pred_grad_x = pred_grad_x[:, :, :grad_H, :grad_W]
    target_grad_x = target_grad_x[:, :, :grad_H, :grad_W]
    pred_grad_y = pred_grad_y[:, :, :grad_H, :grad_W]
    target_grad_y = target_grad_y[:, :, :grad_H, :grad_W]

    return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x_gray, y_rgb in dataloader:
        x_gray, y_rgb = x_gray.to(DEVICE), y_rgb.to(DEVICE)
        output = F.interpolate(model(x_gray), size=(64, 64), mode='bilinear', align_corners=False)


        mse = F.mse_loss(output, y_rgb)
        edge = edge_loss(output, y_rgb)
        loss = mse + EDGE_WEIGHT * edge

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {total_loss:.2f}")

    # Save predictions
    if (epoch + 1) % SAMPLE_EVERY == 0 or epoch == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            val_input = x_gray[:8]
            val_output = model(val_input)

            save_image(val_input, f"samples/cond1/input_gray_epoch{epoch+1}.png", nrow=4)
            save_image(val_output, f"samples/cond1/output_color_epoch{epoch+1}.png", nrow=4)

# Save model
torch.save(model.state_dict(), "models/color_mapper.pth")
print("✅ ColorMapper trained and saved.")
