"""
Deep learning model: Convolutional VAE for level generation.

Improvements over old project:
- Uses all 8 tile categories (not just 3)
- Full 20x40 grid (not cropped to 15x40)
- Larger latent space (64-dim)
- Residual connections in encoder/decoder
- Class-weighted loss to handle imbalanced categories
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data_utils import (
    load_levels, train_test_split_levels, get_model_path,
    NUM_CATEGORIES, GRID_HEIGHT, GRID_WIDTH,
)


class LevelDataset(Dataset):
    def __init__(self, levels):
        self.levels = levels

    def __len__(self):
        return len(self.levels)

    def __getitem__(self, idx):
        level = self.levels[idx]
        one_hot = np.eye(NUM_CATEGORIES, dtype=np.float32)[level]  # (H, W, C)
        one_hot = one_hot.transpose(2, 0, 1)  # (C, H, W)
        return torch.tensor(one_hot), torch.tensor(level, dtype=torch.long)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: (C, 20, 40) -> (256, 5, 10)
        self.encoder = nn.Sequential(
            nn.Conv2d(NUM_CATEGORIES, 64, 3, stride=2, padding=1),  # -> (64, 10, 20)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # -> (128, 5, 10)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # -> (256, 5, 10)
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 5 * 10, latent_dim)
        self.fc_logvar = nn.Linear(256 * 5 * 10, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 5 * 10)
        self.decoder = nn.Sequential(
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),  # -> (128, 5, 10)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # -> (64, 10, 20)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, NUM_CATEGORIES, 3, stride=2, padding=1, output_padding=1),  # -> (8, 20, 40)
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_decode(z))
        h = h.view(h.size(0), 256, 5, 10)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def compute_class_weights(levels):
    """Sqrt-inverse frequency weighting to gently boost rare categories."""
    counts = np.bincount(levels.flatten(), minlength=NUM_CATEGORIES).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    freqs = counts / counts.sum()
    weights = 1.0 / np.sqrt(freqs)
    weights /= weights.mean()
    return torch.tensor(weights)


def vae_loss(recon_logits, target, mu, logvar, class_weights, beta=0.1):
    ce = F.cross_entropy(recon_logits, target, weight=class_weights, reduction="mean")
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return ce + beta * kl, ce, kl


def train_vae(levels, epochs=150, batch_size=32, lr=1e-3, beta=0.1, latent_dim=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, val_data = train_test_split_levels(levels)
    train_dataset = LevelDataset(train_data)
    val_dataset = LevelDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    class_weights = compute_class_weights(train_data).to(device)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            recon, mu, logvar = model(x)
            loss, ce, kl = vae_loss(recon, target, mu, logvar, class_weights, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, target in val_loader:
                x, target = x.to(device), target.to(device)
                recon, mu, logvar = model(x)
                loss, _, _ = vae_loss(recon, target, mu, logvar, class_weights, beta)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), get_model_path("vae_best.pth"))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    model.load_state_dict(torch.load(get_model_path("vae_best.pth"), weights_only=True))
    return model


def generate_levels(model, n=1, temperature=1.0, seed=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        z = torch.randn(n, model.latent_dim).to(device)
        logits = model.decode(z)
        probs = F.softmax(logits / temperature, dim=1)
        # Sample from categorical distribution per pixel
        b, c, h, w = probs.shape
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, c)
        samples = torch.multinomial(probs_flat, 1).reshape(b, h, w)

    return samples.cpu().numpy()


if __name__ == "__main__":
    levels = load_levels()
    print(f"Training on {len(levels)} levels...")

    model = train_vae(levels, epochs=150, batch_size=32, lr=1e-3, beta=0.1, latent_dim=64)

    sample = generate_levels(model, n=1, seed=42)[0]
    print(f"Sample category counts: {dict(zip(*np.unique(sample, return_counts=True)))}")
