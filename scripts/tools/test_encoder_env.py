#!/usr/bin/env python3
"""Quick test: encoder env creation + encoder loading."""
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
app.app

import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

cfg = parse_env_cfg("Isaac-Factory-PegInsert-Encoder-Direct-v0", device="cuda:0", num_envs=1)
env = gym.make("Isaac-Factory-PegInsert-Encoder-Direct-v0", cfg=cfg)
obs, _ = env.reset()
print(f"policy={obs['policy'].shape}  critic={obs['critic'].shape}")
print(f"Expected: policy=(1, 282)  critic=(1, 305)")
env.close()
print("OK")
