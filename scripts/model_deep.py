"""
Conditional Convolutional VAE for platformer level generation.

Changes vs. the earlier unconditional version:
- Trains on 3 gameplay classes (empty / solid / hazard) — matches the
  in-browser game's physics, removes wasted capacity on visual-only
  categories (bonus/water/decoration/slope/platform all collapse to
  their gameplay equivalent).
- Decoder is conditioned on a difficulty bucket (tertile of a structural
  difficulty score — see scripts/difficulty.py). The bucket one-hot is
  concatenated onto the latent z, so generation can be steered by the
  UI difficulty slider.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data_utils import (
    load_levels, train_val_test_split_levels, get_model_path,
    GRID_HEIGHT, GRID_WIDTH,
)
from difficulty import assign_buckets
from repair import enforce_layout


NUM_CLASSES = 3  # 0=empty, 1=solid, 2=hazard
N_BUCKETS = 3


def pick_device():
    """Prefer CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_three_class(levels):
    """
    Collapse 8-category levels to the 3 gameplay categories the game actually
    uses. Walkable (solid/slope/platform) → 1; hazard → 2; everything else
    (empty/bonus/water/decoration) → 0.
    """
    out = np.zeros_like(levels)
    out[np.isin(levels, (1, 2, 3))] = 1
    out[levels == 6] = 2
    return out


class LevelDataset(Dataset):
    def __init__(self, levels_3c, buckets):
        assert len(levels_3c) == len(buckets)
        self.levels = levels_3c
        self.buckets = buckets

    def __len__(self):
        return len(self.levels)

    def __getitem__(self, idx):
        level = self.levels[idx]
        one_hot = np.eye(NUM_CLASSES, dtype=np.float32)[level]
        one_hot = one_hot.transpose(2, 0, 1)
        return (
            torch.tensor(one_hot),
            torch.tensor(level, dtype=torch.long),
            torch.tensor(self.buckets[idx], dtype=torch.long),
        )


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
    def __init__(self, latent_dim=64, n_classes=NUM_CLASSES, n_buckets=N_BUCKETS):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.n_buckets = n_buckets

        self.encoder = nn.Sequential(
            nn.Conv2d(n_classes, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 5 * 10, latent_dim)
        self.fc_logvar = nn.Linear(256 * 5 * 10, latent_dim)

        self.fc_decode = nn.Linear(latent_dim + n_buckets, 256 * 5 * 10)
        self.decoder = nn.Sequential(
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, n_classes, 3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, bucket):
        """bucket: (B,) long or (B, n_buckets) float. Conditioning is
        concatenated onto z before the decoder projection."""
        if bucket.dim() == 1:
            cond = F.one_hot(bucket, num_classes=self.n_buckets).float()
        else:
            cond = bucket.float()
        zc = torch.cat([z, cond], dim=1)
        h = F.relu(self.fc_decode(zc))
        h = h.view(h.size(0), 256, 5, 10)
        return self.decoder(h)

    def forward(self, x, bucket):
        mu, logvar = self.encode(x)
        # Clamp logvar so exp(logvar) can't overflow during KL/sampling.
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, bucket)
        return recon, mu, logvar


def compute_class_weights(levels_3c, max_weight=1.5):
    counts = np.bincount(levels_3c.flatten(), minlength=NUM_CLASSES).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    freqs = counts / counts.sum()
    weights = 1.0 / np.sqrt(freqs)
    weights /= weights.mean()
    # Cap the hazard-class weight so the model doesn't over-generate hazards
    # in the easy bucket. With the default sqrt-inverse scheme hazard ≈ 2.25,
    # which was bleeding hazards into bucket 0 at inference.
    weights = np.minimum(weights, max_weight)
    return torch.tensor(weights)


def vae_loss(recon_logits, target, mu, logvar, class_weights, beta=0.1):
    ce = F.cross_entropy(recon_logits, target, weight=class_weights, reduction="mean")
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return ce + beta * kl, ce, kl


def train_vae(levels_8c, epochs=150, batch_size=64, lr=1e-3, beta=0.1, latent_dim=64):
    device = pick_device()
    print(f"training on {device}")

    # Bucket on the full set, then drop chunks with no reachable start so
    # the VAE doesn't learn a noisy 'degenerate-chunk' mode in any bucket.
    buckets, valid_mask, info = assign_buckets(levels_8c, n_buckets=N_BUCKETS)
    levels_8c = levels_8c[valid_mask]
    buckets = buckets[valid_mask]
    print(f"kept {len(levels_8c)}/{info['n_valid'] + info['n_dropped']} chunks "
          f"({info['n_dropped']} dropped: no reachable start)")
    levels_3c = to_three_class(levels_8c)
    # Bake the layout constraint into the training data so the VAE doesn't
    # spend capacity modeling the top-sky / bottom-floor bands that are
    # enforced at inference time anyway.
    levels_3c = np.stack([enforce_layout(l) for l in levels_3c])

    # 3-way split: train / val / test (80/10/10). Val drives checkpoint
    # selection; test is held out for final evaluation only (evaluate.py).
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(levels_3c))
    n = len(levels_3c)
    n_test = int(n * 0.10)
    n_val = int(n * 0.10)
    n_train = n - n_val - n_test
    tr_idx = indices[:n_train]
    va_idx = indices[n_train:n_train + n_val]

    train_dataset = LevelDataset(levels_3c[tr_idx], buckets[tr_idx])
    val_dataset = LevelDataset(levels_3c[va_idx], buckets[va_idx])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    class_weights = compute_class_weights(levels_3c[tr_idx]).to(device)

    print(f"bucket counts (train): {np.bincount(buckets[tr_idx]).tolist()}")
    print(f"class weights: {class_weights.tolist()}")

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, target, bucket in train_loader:
            x, target, bucket = x.to(device), target.to(device), bucket.to(device)
            recon, mu, logvar = model(x, bucket)
            loss, ce, kl = vae_loss(recon, target, mu, logvar, class_weights, beta)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, target, bucket in val_loader:
                x, target, bucket = x.to(device), target.to(device), bucket.to(device)
                recon, mu, logvar = model(x, bucket)
                loss, _, _ = vae_loss(recon, target, mu, logvar, class_weights, beta)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), get_model_path("cvae_best.pth"))

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    model.load_state_dict(torch.load(get_model_path("cvae_best.pth"), weights_only=True))
    return model, info


def generate_levels(model, n=1, bucket=1, temperature=1.0, seed=None, device=None):
    """
    bucket: int in [0, n_buckets-1] (easy/med/hard) — required conditioning.
    Returns 3-class arrays (values in {0, 1, 2}).
    """
    if device is None:
        device = pick_device()
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        z = torch.randn(n, model.latent_dim).to(device)
        bucket_ids = torch.full((n,), int(bucket), dtype=torch.long, device=device)
        logits = model.decode(z, bucket_ids)
        probs = F.softmax(logits / temperature, dim=1)
        b, c, h, w = probs.shape
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, c)
        samples = torch.multinomial(probs_flat, 1).reshape(b, h, w)

    return samples.cpu().numpy()


def three_class_to_api(levels_3c):
    """Map 3-class labels {0,1,2} back to the API schema {0=empty, 1=solid, 6=hazard}."""
    out = np.asarray(levels_3c).copy()
    out[out == 2] = 6
    return out


if __name__ == "__main__":
    levels = load_levels()
    print(f"Training on {len(levels)} levels...")

    model, info = train_vae(levels, epochs=150, batch_size=64, lr=1e-3, beta=0.1, latent_dim=64)

    for b in range(N_BUCKETS):
        sample = generate_levels(model, n=1, bucket=b, seed=42)[0]
        counts = dict(zip(*np.unique(sample, return_counts=True)))
        print(f"bucket={b} sample counts: {counts}")
