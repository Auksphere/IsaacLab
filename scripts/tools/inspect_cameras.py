#!/usr/bin/env python3
"""
Quick camera inspection for Isaac-Factory-PegInsert-Vision-Direct-v0.

Saves left/right TiledCamera frames as PNGs, then compiles to mp4 via ffmpeg.

Usage (inside container):
  # Quick: 30 random-action frames
  ./isaaclab.sh -p scripts/tools/inspect_cameras.py \
    --num-frames 30 --output-dir /tmp/camera_check \
    --headless --enable_cameras

  # Expert rollout (with checkpoint):
  ./isaaclab.sh -p scripts/tools/inspect_cameras.py \
    --checkpoint /workspace/isaaclab/logs/rl_games/Factory/peg_insert_ep73/nn/Factory.pth \
    --num-frames 150 --output-dir /workspace/isaaclab/data/camera_check \
    --headless --enable_cameras
"""
import argparse, math, os, subprocess, sys, time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect camera streams from vision env.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Optional RL Games checkpoint for expert rollout.")
parser.add_argument("--num-frames", type=int, default=30,
                    help="Number of frames to capture (default: 30).")
parser.add_argument("--output-dir", type=str, default="/tmp/camera_check",
                    help="Output directory for PNG frames + video.")
parser.add_argument("--fps", type=int, default=15,
                    help="Video frame rate (default: 15, matches policy freq).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── All Isaac Lab imports AFTER AppLauncher ──
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from PIL import Image

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.direct.factory.factory_env_cfg import (
    FactoryTaskPegInsertCfg,
    FactoryTaskPegInsertVisionCfg,
)

# ── Create vision env with base obs_order for checkpoint compatibility ──
print("[INFO] Creating vision env (1 env, base obs_order for ckpt compat)...")
cfg = parse_env_cfg("Isaac-Factory-PegInsert-Vision-Direct-v0", device="cuda:0", num_envs=1)

# Override obs/state order to match BASE training env so checkpoint loads correctly.
base_cfg = FactoryTaskPegInsertCfg()
cfg.obs_order = list(base_cfg.obs_order)
cfg.state_order = list(base_cfg.state_order)
cfg.seed = 42

env = gym.make("Isaac-Factory-PegInsert-Vision-Direct-v0", cfg=cfg)

# ── Load checkpoint if provided ──
player = None
if args_cli.checkpoint is not None:
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner
    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")

    # Load agent config from YAML (same as collect_unified_dataset.py)
    import yaml
    agent_yaml = (
        Path(__file__).resolve().parents[2] / "source" / "isaaclab_tasks"
        / "isaaclab_tasks" / "direct" / "factory" / "agents" / "rl_games_ppo_cfg.yaml"
    )
    with open(agent_yaml) as f:
        agent_cfg = yaml.safe_load(f)

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = os.path.abspath(args_cli.checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = 1
    agent_cfg["params"]["seed"] = 42

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    venv = RlGamesVecEnvWrapper(env, "cuda:0", clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda cn, na, **kw: type('X', (), {
            '__init__': lambda s: None,
            'get_number_of_agents': lambda s: 1,
        })(),
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: venv}
    )

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(agent_cfg["params"]["load_path"])
    player.has_batch_dimension = True
    print("[INFO] Checkpoint loaded successfully.")
else:
    print("[INFO] No checkpoint — using zero actions (robot holds still).")

# ── Output directory ──
output_dir = Path(args_cli.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ── Main capture loop ──
print(f"[INFO] Capturing {args_cli.num_frames} frames...")
frames_left = []
frames_right = []

obs_dict, _ = env.reset()
if player is not None:
    player.reset()

for step in range(args_cli.num_frames):
    # ── Save camera images from current observation ──
    if "camera_left_rgb" in obs_dict:
        left_rgb = obs_dict["camera_left_rgb"][0].cpu().numpy()  # (224,224,3) uint8
        right_rgb = obs_dict["camera_right_rgb"][0].cpu().numpy()

        left_path = output_dir / f"step_{step:04d}_L.png"
        right_path = output_dir / f"step_{step:04d}_R.png"
        Image.fromarray(left_rgb).save(left_path)
        Image.fromarray(right_rgb).save(right_path)

        frames_left.append(left_rgb)
        frames_right.append(right_rgb)
    else:
        print(f"[WARN] No camera data in obs_dict at step {step}! "
              "Did you pass --enable_cameras?")
        break

    # ── Get action and step ──
    if player is not None:
        obs_tensor = obs_dict["policy"]
        action = player.get_action(obs_tensor, is_deterministic=True)
    else:
        action = torch.zeros(1, 6, device="cuda:0")

    obs_dict, rew, term, trunc, info = env.step(action)

    if term.any() or trunc.any():
        print(f"[INFO] Episode ended at step {step}")
        break

    if (step + 1) % 50 == 0:
        print(f"  ... {step + 1}/{args_cli.num_frames} frames captured")

env.close()
n = len(frames_left)
print(f"[INFO] Saved {n} frame pairs to {output_dir}/")

# ── Compile to video ──
# Try imageio first (bundled ffmpeg), then system ffmpeg, then give instructions.
video_ok = False

# Method 1: imageio (ships its own ffmpeg binary via imageio-ffmpeg)
try:
    import imageio
    for side, frames in [("left", frames_left), ("right", frames_right)]:
        video_path = output_dir / f"camera_{side}.mp4"
        writer = imageio.get_writer(str(video_path), fps=args_cli.fps, codec="libx264", quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"[INFO] Video (imageio): {video_path}")
    video_ok = True
except ImportError:
    pass

# Method 2: system ffmpeg
if not video_ok:
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        for side, letter in [("left", "L"), ("right", "R")]:
            video_path = str(output_dir / f"camera_{side}.mp4")
            ret = subprocess.run(
                [ffmpeg, "-y", "-framerate", str(args_cli.fps),
                 "-i", str(output_dir / f"step_%04d_{letter}.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path],
                capture_output=True, text=True,
            )
            if ret.returncode == 0:
                print(f"[INFO] Video (ffmpeg): {video_path}")
                video_ok = True
            else:
                print(f"[WARN] ffmpeg failed: {ret.stderr[:300]}")

if not video_ok:
    print(f"[INFO] No video encoder found in container. "
          f"Copy PNGs to host and compile manually:")
    for side in ["left", "right"]:
        print(f"  ffmpeg -framerate {args_cli.fps} -i {output_dir}/step_%04d_{side[0].upper()}.png "
              f"-c:v libx264 -pix_fmt yuv420p {output_dir}/camera_{side}.mp4")

print("[INFO] Done.")
simulation_app.close()
