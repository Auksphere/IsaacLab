#!/usr/bin/env python3
"""Quick check: are camera images valid in the encoder env?"""
import argparse, time
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks
import numpy as np
from PIL import Image
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

cfg = parse_env_cfg("Isaac-Factory-PegInsert-Encoder-Direct-v0", device="cuda:0", num_envs=1)
env = gym.make("Isaac-Factory-PegInsert-Encoder-Direct-v0", cfg=cfg)

obs, _ = env.reset()
for step in range(5):
    action = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(action)

    if "camera_left_rgb" in obs:
        img = obs["camera_left_rgb"][0].cpu().numpy()
        out = f"/workspace/isaaclab/scripts/tools/cam_check_step{step}.png"
        Image.fromarray(img).save(out)
        print(f"Step {step}: saved {out}  min={img.min()} max={img.max()} mean={img.mean():.0f}")

env.close()
simulation_app.close()
