# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import gymnasium as gym

from . import agents
from .factory_env import FactoryEnv
from .factory_env_cfg import (
    FactoryTaskGearMeshCfg,
    FactoryTaskGearMeshEncoderCfg,
    FactoryTaskNutThreadCfg,
    FactoryTaskPegInsertCfg,
    FactoryTaskPegInsertVisionCfg,
    FactoryTaskPegInsertEncoderCfg,
    FactoryTaskPegInsertEncoderCfg,
    FactoryTaskPegInsertMonoEncoderCfg,
    FactoryTaskUSBInsertEncoderCfg,
)

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Factory-PegInsert-Direct-v0",
    entry_point="isaaclab_tasks.direct.factory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-GearMesh-Direct-v0",
    entry_point="isaaclab_tasks.direct.factory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-NutThread-Direct-v0",
    entry_point="isaaclab_tasks.direct.factory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-PegInsert-Vision-Direct-v0",
    entry_point="isaaclab_tasks.direct.factory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertVisionCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-GearMesh-Encoder-Direct-v0",
    entry_point="isaaclab_tasks.direct.factory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskGearMeshEncoderCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_encoder_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-PegInsert-Mono-Encoder-Direct-v0",
    entry_point="isaaclab_tasks.direct.factory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertMonoEncoderCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_encoder_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-PegInsert-Encoder-Direct-v0",
    entry_point="isaaclab_tasks.direct.factory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertEncoderCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_encoder_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
)

gym.register(
    id="Isaac-Factory-USBInsert-Encoder-Direct-v0",
    entry_point="isaaclab_tasks.direct.factory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskUSBInsertEncoderCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_encoder_cfg.yaml",
    },
)
