#!/usr/bin/env python3
"""Train simple BC MLP: obs(285) → action(6) on HDF5 dataset.

Matches RL Games actor_mlp architecture for easy weight loading.
Output: bc_mlp.pt

Usage (host machine):
  python train_bc_mlp.py --data ../../output/bc_train.hdf5 --epochs 200
"""

import argparse, os, sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load HDF5 data
    hf = h5py.File(args.data, "r")
    all_obs, all_act = [], []
    for demo_name in tqdm(hf["data"].keys(), desc="Loading data"):
        demo = hf["data"][demo_name]
        obs = demo["obs/flat_obs"][:]  # (T, 285)
        act = demo["actions"][:]        # (T, 6)
        all_obs.append(torch.from_numpy(obs.astype(np.float32)))
        all_act.append(torch.from_numpy(act.astype(np.float32)))
    hf.close()

    X = torch.cat(all_obs, dim=0)
    Y = torch.cat(all_act, dim=0)
    print(f"Data: {X.shape[0]} frames, obs={X.shape[1]}D, act={Y.shape[1]}D")

    # Normalize inputs → register as buffers (saved in state_dict)
    x_mean = X.mean(dim=0, keepdim=True)
    x_std = X.std(dim=0, keepdim=True).clamp(min=1e-4)
    print(f"Obs: mean={x_mean.mean().item():.3f}, std={x_std.mean().item():.3f}")

    # Split train/val
    n_val = min(int(len(X) * 0.1), 1000)
    perm = torch.randperm(len(X))
    X_train, Y_train = X[perm[n_val:]], Y[perm[n_val:]]
    X_val, Y_val = X[perm[:n_val]], Y[perm[:n_val]]

    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    # Architecture: matches RL Games actor_mlp (285→512→128→64) + mu (64→6)
    class BCMLP(nn.Sequential):
        def __init__(self):
            super().__init__(
                nn.Linear(285, 512), nn.ELU(),
                nn.Linear(512, 128), nn.ELU(),
                nn.Linear(128, 64), nn.ELU(),
                nn.Linear(64, 6),
            )
            self.register_buffer("x_mean", x_mean.squeeze(0))
            self.register_buffer("x_std", x_std.squeeze(0))

    model = BCMLP().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            xb = (xb - model.x_mean.unsqueeze(0)) / model.x_std.unsqueeze(0)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                xb = (xb - model.x_mean.unsqueeze(0)) / model.x_std.unsqueeze(0)
                pred = model(xb)
                val_loss += F.mse_loss(pred, yb).item()
        val_loss /= len(val_dl)

        scheduler.step(val_loss)

        if epoch % 20 == 0 or val_loss < best_val:
            if val_loss < best_val:
                best_val = val_loss
                output_path = args.output or str(Path(args.data).parent / "bc_mlp.pt")
                torch.save(model.state_dict(), output_path)
                print(f"  → saved (val_loss={val_loss:.6f})")

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    print(f"\nBest val_loss={best_val:.6f}")
    print(f"Model saved to {output_path}")
    print("Action MSE (sqrt):", np.sqrt(best_val))


if __name__ == "__main__":
    main()
