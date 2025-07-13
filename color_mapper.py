import torch
import torch.nn as nn

class ColorMapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # → (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, stride=2, padding=1), # → (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),          # → (128, 16, 16)
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # → (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # → (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 3, 1),          # Final RGB output (3, 64, 64)
            nn.Sigmoid()                 # Scale to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x
