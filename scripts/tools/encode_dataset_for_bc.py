#!/usr/bin/env python3
"""Encode demo dataset using frozen encoder → HDF5 format for BC/LfD.

Reads JSONL+PNG dataset, runs encoder to produce 282D observations,
writes HDF5 files compatible with DemoBuffer for PPO demo mixing.

Usage (host machine, no Isaac Sim):
  python encode_dataset_for_bc.py \
    --encoder-ckpt ../../output/pretrain/pretrained_encoder_best.pt \
    --data-dir ../../data --output ../../output/bc_train.hdf5
"""

import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent / "source" / "isaaclab_tasks"
                       / "isaaclab_tasks" / "direct" / "factory"))
from encoder import VisualForceEncoder, preprocess_rgb, IMG_SIZE

FT_FRAMES = 10
FT_DIM = 6
K = 3


def load_jsonl(jsonl_path):
    """Load all episodes from JSONL, return dict[ep_id -> list[frame_dict]]."""
    episodes = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            episodes[rec["episode_id"]].append(rec)
    # Sort each episode by frame_idx
    for ep_id in episodes:
        episodes[ep_id].sort(key=lambda r: r["frame_idx"])
    return episodes


def build_obs_for_frame(ep, idx, prev_action_6d, encoder, device):
    """Build 282D observation for a single frame.

    Returns: obs_282 (np.ndarray of shape (282,))
    obs = z(256) + proprio(20) + prev_a(6) — matches RL policy input
    """
    # Image: load from disk, preprocess
    img_dir = Path(args.data_dir) / args.split / "images"
    l_path = img_dir / Path(ep[idx]["camera_left"]).name
    r_path = img_dir / Path(ep[idx]["camera_right"]).name

    img_l = np.array(Image.open(str(l_path)).convert("RGB"), dtype=np.float32)
    img_r = np.array(Image.open(str(r_path)).convert("RGB"), dtype=np.float32)
    img_l = torch.from_numpy(img_l / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    img_r = torch.from_numpy(img_r / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    if img_l.shape[-2] != IMG_SIZE:
        img_l = F.interpolate(img_l, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        img_r = F.interpolate(img_r, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_l = (img_l - IMAGENET_MEAN) / IMAGENET_STD
    img_r = (img_r - IMAGENET_MEAN) / IMAGENET_STD

    # Force: 10-frame window ending at idx
    def get_ft(i):
        ft = ep[max(0, min(i, len(ep) - 1))].get("force_torque", [0.0] * FT_DIM)
        return [float(v) for v in ft[:FT_DIM]]

    ft_3f = []
    for offset in range(-(FT_FRAMES - 1), 1):
        ft_3f.extend(get_ft(idx + offset))
    ft_3f = torch.tensor(ft_3f, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 60)

    idx_k = idx - K
    ft_3f_k = []
    for offset in range(-(FT_FRAMES - 1), 1):
        ft_3f_k.extend(get_ft(max(0, idx_k + offset)))
    ft_3f_k = torch.tensor(ft_3f_k, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 60)

    # Proprio
    rec = ep[idx]
    joint_pos = [float(v) for v in rec.get("joint_pos", [0] * 7)[:7]]
    fingertip_pos = [float(v) for v in rec.get("fingertip_pos", [0] * 3)[:3]]
    fingertip_quat = [float(v) for v in rec.get("fingertip_quat", [0] * 4)[:4]]
    ee_linvel = [float(v) for v in rec.get("ee_linvel", [0] * 3)[:3]]
    ee_angvel = [float(v) for v in rec.get("ee_angvel", [0] * 3)[:3]]
    proprio_20 = joint_pos + fingertip_pos + fingertip_quat + ee_linvel + ee_angvel
    proprio_20 = torch.tensor(proprio_20, dtype=torch.float32, device=device).unsqueeze(0)

    # Encode → z_vis(128) + z_force(128) = 256D features, + task_norm(3)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            z_vis, z_force, _, _ = encoder.forward_features(
                img_l, img_r, ft_3f, ft_3f_k,
                prev_action_6d.unsqueeze(0), proprio_20)

    z = torch.cat([z_vis, z_force], dim=-1).float().cpu().squeeze(0)  # (256,)
    proprio = proprio_20.float().cpu().squeeze(0)   # (20,)
    prev_a = prev_action_6d.float().cpu()           # (6,)

    # RL policy obs: z(256) + proprio(20) + prev_a(6) = 282D
    obs = torch.cat([z, proprio, prev_a], dim=-1)
    return obs.numpy()


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-ckpt", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="dinov2_vits14_attn")
    parser.add_argument("--data-dir", type=str, default=str(THIS_DIR.parent.parent / "data"))
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    split = args.split

    # Load encoder
    print(f"Loading encoder: {args.encoder_ckpt}")
    encoder = VisualForceEncoder.from_checkpoint(args.encoder_ckpt, args.backbone, device)
    encoder.eval()
    print(f"  feature_dim={encoder.feature_dim}")

    # Load dataset
    jsonl_path = data_dir / split / "labels.jsonl"
    print(f"Loading dataset: {jsonl_path}")
    episodes = load_jsonl(jsonl_path)
    print(f"  {len(episodes)} episodes")

    # Output HDF5 → isaaclab/output/
    output_dir = THIS_DIR.parent.parent / "output"
    output_path = args.output or str(output_dir / f"bc_{split}.hdf5")
    print(f"Writing HDF5: {output_path}")
    hf = h5py.File(output_path, "w")
    data_group = hf.create_group("data")
    data_group.attrs["total"] = 0
    data_group.attrs["env_args"] = json.dumps({
        "env_name": "Isaac-Factory-PegInsert-Encoder-Direct-v0",
        "type": 2  # robomimic gym env type
    })

    total_frames = 0
    demo_idx = 0

    for ep_id, ep in tqdm(sorted(episodes.items()), desc="Encoding episodes"):
        frames = len(ep)
        if frames < FT_FRAMES + K:
            continue

        obs_list = []
        action_list = []
        prev_action = torch.zeros(6, device=device)  # first step: zero action

        for t in range(frames):
            if t < FT_FRAMES - 1 or t - K < FT_FRAMES - 1:
                # Not enough history for 3-frame window → skip
                # But still need to track prev_action for next step
                rec = ep[max(0, min(t, frames - 1))]
                a = rec.get("action", [0.0] * 6)
                prev_action = torch.tensor([float(v) for v in a[:6]], device=device)
                continue

            obs = build_obs_for_frame(ep, t, prev_action, encoder, device)
            obs_list.append(obs)

            rec = ep[t]
            a = rec.get("action", [0.0] * 6)
            action_6d = torch.tensor([float(v) for v in a[:6]], device=device)
            action_list.append(action_6d.cpu().numpy())
            prev_action = action_6d

        if len(obs_list) < 10:
            continue  # skip very short usable episodes

        obs_arr = np.stack(obs_list, axis=0)    # (T, 285)
        act_arr = np.stack(action_list, axis=0)  # (T, 6)

        # Create demo group
        demo_group = data_group.create_group(f"demo_{demo_idx}")
        demo_group.attrs["num_samples"] = len(obs_arr)
        demo_group.attrs["seed"] = ep_id

        # Determine success from last frame keypoint_dist
        last_kd = ep[-1].get("keypoint_dist", 1.0)
        demo_group.attrs["success"] = bool(last_kd < 0.002)

        # Write observations and actions
        obs_subgroup = demo_group.create_group("obs")
        obs_subgroup.create_dataset("flat_obs", data=obs_arr)
        demo_group.create_dataset("actions", data=act_arr)

        total_frames += len(obs_arr)
        demo_idx += 1

    data_group.attrs["total"] = total_frames
    hf.close()

    print(f"\nDone: {demo_idx} episodes, {total_frames} frames → {output_path}")
    print(f"Observation dim: 282 = z(256)+proprio(20)+prev_a(6)")


if __name__ == "__main__":
    main()
