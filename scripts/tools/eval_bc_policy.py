#!/usr/bin/env python3
"""Evaluate BC policy success rate in the encoder env.

Loads frozen encoder + trained BC (robomimic RNN) policy,
runs deterministic rollout with camera rendering.

Usage:
  ./isaaclab.sh -p scripts/tools/eval_bc_policy.py \
    --encoder-ckpt /workspace/isaaclab/output/pretrain/pretrained_encoder_best.pt \
    --bc-ckpt /workspace/isaaclab/logs/robomimic/.../models/model_epoch_400.pth \
    --episodes 20 --headless --enable_cameras
"""

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--encoder-ckpt", type=str, required=True)
parser.add_argument("--bc-ckpt", type=str, default="")   # robomimic checkpoint
parser.add_argument("--bc-mlp", type=str, default="")     # simple BC MLP checkpoint
parser.add_argument("--backbone", type=str, default="dinov2_vits14")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--max-steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent / "source" / "isaaclab_tasks"
                       / "isaaclab_tasks" / "direct" / "factory"))
from encoder import VisualForceEncoder, preprocess_rgb

# robomimic imports (available inside Isaac Sim Docker) — lazy import


def main():
    device = "cuda:0"
    N = 1

    # ── Load encoder ──
    print(f"[INFO] Loading encoder: {args_cli.encoder_ckpt}")
    encoder = VisualForceEncoder.from_checkpoint(args_cli.encoder_ckpt, args_cli.backbone, device)
    encoder.eval()
    print(f"  feature_dim={encoder.feature_dim}")

    # ── Load BC policy ──
    bc_mlp = None
    bc_mean = bc_std = None
    use_robomimic = args_cli.bc_ckpt and args_cli.bc_ckpt != ""
    if use_robomimic:
        from robomimic.utils.file_utils import policy_from_checkpoint
        print(f"[INFO] Loading BC robomimic policy: {args_cli.bc_ckpt}")
        bc_policy, _ = policy_from_checkpoint(ckpt_path=args_cli.bc_ckpt, device=device)
        print(f"  BC policy loaded (robomimic)")
    elif args_cli.bc_mlp and args_cli.bc_mlp != "":
        print(f"[INFO] Loading BC MLP: {args_cli.bc_mlp}")
        bc_state = torch.load(args_cli.bc_mlp, map_location=device, weights_only=True)
        bc_mlp = torch.nn.Sequential(
            torch.nn.Linear(285, 512), torch.nn.ELU(),
            torch.nn.Linear(512, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 64), torch.nn.ELU(),
            torch.nn.Linear(64, 6),
        ).to(device)
        bc_mlp.load_state_dict(bc_state, strict=False)
        bc_mlp.eval()
        bc_mean = bc_state["x_mean"].to(device)
        bc_std = bc_state["x_std"].to(device)
        print(f"  BC MLP loaded ({len(bc_state)-2} weights)")
    else:
        raise ValueError("Either --bc-ckpt or --bc-mlp must be provided")

    # ── Create env with cameras for encoder ──
    print(f"[INFO] Creating env ({N} envs, cameras ON)...")
    cfg = parse_env_cfg("Isaac-Factory-PegInsert-Encoder-Direct-v0", device=device, num_envs=N)
    cfg.encoder_debug_state_policy = True     # 43D privileged obs (unused by BC)
    cfg.encoder_debug_keep_cameras = True     # keep cameras for encoder
    cfg.encoder_checkpoint = ""               # don't auto-load encoder
    cfg.seed = 42
    env = gym.make("Isaac-Factory-PegInsert-Encoder-Direct-v0", cfg=cfg)
    env_ = env.unwrapped

    # ── Evaluation loop ──
    results = []
    all_rewards = []

    for ep in range(args_cli.episodes):
        obs_dict, _ = env.reset()
        if use_robomimic:
            bc_policy.start_episode()  # reset BC LSTM state
        prev_action = torch.zeros(N, 6, device=device)
        ep_reward = 0.0
        ep_frames = 0
        ep_min_kd = 1.0
        success = False
        ep_rew_list = []

        # Warmup step (ensures camera renders first frame)
        obs_dict, rew, term, trunc, _ = env.step(torch.zeros(N, 6, device=device))

        for step in range(args_cli.max_steps):
            # ── Encode observation ──
            img_l = preprocess_rgb(env_._tiled_camera_left.data.output["rgb"])
            img_r = preprocess_rgb(env_._tiled_camera_right.data.output["rgb"])
            ft_3f = env_.ft_ring[:, -3:, :].reshape(N, 18)
            ft_3f_k = env_.ft_ring[:, :3, :].reshape(N, 18)
            proprio_20 = torch.cat([
                env_.joint_pos[:, 0:7], env_.fingertip_midpoint_pos,
                env_.fingertip_midpoint_quat, env_.ee_linvel_fd, env_.ee_angvel_fd,
            ], dim=-1)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    z, task_pred = encoder.forward_features(
                        img_l, img_r, ft_3f, ft_3f_k, prev_action, proprio_20)

            # Build 285D obs
            obs_285 = torch.cat([
                z.float(), task_pred.float(), proprio_20.float(), prev_action.float()
            ], dim=-1)  # (1, 285)

            # ── BC policy inference ──
            if use_robomimic:
                obs_np = obs_285.cpu().numpy().squeeze(0)  # (285,)
                bc_action = bc_policy({"flat_obs": obs_np})  # (6,) numpy
                actions = torch.from_numpy(bc_action).to(device).unsqueeze(0)
            else:
                bc_obs = (obs_285 - bc_mean) / bc_std
                actions = bc_mlp(bc_obs)  # (1, 6)

            # ── Step env ──
            obs_dict, rew, term, trunc, info = env.step(actions)
            prev_action = actions.clone()

            ep_reward += rew[0].item()
            ep_frames += 1
            ep_rew_list.append(rew[0].item())

            kd = env_.keypoint_dist[0].item()
            ep_min_kd = min(ep_min_kd, kd)

            if step % 25 == 0:
                print(f"  ep={ep:2d} step={step:3d} kd={kd:.4f} min_kd={ep_min_kd:.4f} "
                      f"rew={rew[0].item():.3f} cum_rew={ep_reward:.1f} "
                      f"act_range=[{actions.min().item():.2f}, {actions.max().item():.2f}]")

            if term or trunc:
                success = ep_min_kd < 0.002
                break

        results.append({
            "ep": ep, "frames": ep_frames, "success": success,
            "min_kd": ep_min_kd, "reward": ep_reward,
        })
        all_rewards.append(ep_rew_list)

        print(f"Ep {ep:2d}: succ={success} frames={ep_frames} min_kd={ep_min_kd:.4f} "
              f"cum_rew={ep_reward:.1f}")

    env.close()

    # ── Summary ──
    succ_rate = sum(r["success"] for r in results) / len(results)
    mean_rew = np.mean([r["reward"] for r in results])
    mean_frames = np.mean([r["frames"] for r in results])

    print(f"\n{'='*60}")
    print(f"BC Policy Evaluation ({len(results)} episodes):")
    print(f"  Success rate:  {succ_rate*100:.0f}% ({sum(r['success'] for r in results)}/{len(results)})")
    print(f"  Mean reward:   {mean_rew:.1f}")
    print(f"  Mean frames:   {mean_frames:.0f}")

    min_kds = [r["min_kd"] for r in results]
    print(f"  Min kd range:  [{min(min_kds):.4f}, {max(min_kds):.4f}]")

    if succ_rate >= 0.6:
        print("  ✅ BC policy is viable for residual RL base.")
    elif succ_rate >= 0.3:
        print("  ⚠️  BC partially works — may need more training.")
    else:
        print("  ❌ BC not working well enough.")

    simulation_app.close()


if __name__ == "__main__":
    main()
