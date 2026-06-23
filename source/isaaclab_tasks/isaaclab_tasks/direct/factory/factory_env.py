# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import torch.nn.functional as F

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_apply

from . import factory_control as fc
from .factory_env_cfg import (
    CONTACT_DIM_CFG,
    FORCE_DIM_CFG,
    FT_FILTER_CFG,
    OBS_DIM_CFG,
    STATE_DIM_CFG,
    FactoryEnvCfg,
)


class FactoryEnv(DirectRLEnv):
    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        # Merge DIM_CFGs so that both base and vision configs resolve correctly.
        obs_dim_cfg = {**OBS_DIM_CFG, **FORCE_DIM_CFG, **CONTACT_DIM_CFG}
        state_dim_cfg = {**STATE_DIM_CFG, **FORCE_DIM_CFG, **CONTACT_DIM_CFG}

        cfg.observation_space = sum([obs_dim_cfg[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([state_dim_cfg[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space

        # Stage F encoder mode: override policy/critic obs dims
        encoder_ckpt = getattr(cfg, "encoder_checkpoint", "")
        self._encoder_debug_state_policy = getattr(cfg, "encoder_debug_state_policy", False)
        self._encoder_ablate_bottleneck = getattr(cfg, "encoder_ablate_bottleneck", False)
        print(f"[INIT] encoder_debug_state_policy={self._encoder_debug_state_policy} (FORCED) "
              f"ablate_bottleneck={self._encoder_ablate_bottleneck} "
              f"encoder_ckpt={repr(encoder_ckpt)}")
        self._curriculum_pos_alpha = 0.0
        self._curriculum_pos_alpha_delay = 0
        self._curriculum_pos_alpha_decay_steps = 0
        self._curriculum_pos_alpha_success_threshold = 0.9
        self._curriculum_total_steps = 0
        self._curriculum_decay_progress = 0
        self._curriculum_curr_success_rate = 0.0
        if self._encoder_debug_state_policy:
            # Debug mode: actor sees same privileged state as critic (43D)
            cfg.observation_space = cfg.state_space
            if not getattr(cfg, "encoder_debug_keep_cameras", False):
                cfg.tiled_camera_left = None
                cfg.tiled_camera_right = None
        elif self._encoder_ablate_bottleneck:
            # Ablation: no encoder, κ_hat from privileged keypoint_dist
            # Policy = [κ_hat(1), proprio(20), prev_a(6)] = 27D
            cfg.observation_space = 29  # held_pos_rel_fixed(3) + proprio(20) + prev_a(6)
            cfg.tiled_camera_left = None
            cfg.tiled_camera_right = None
        elif encoder_ckpt:
                # Bottleneck(256D) + proprio(20D) + prev_a(6D) = 282D
                cfg.observation_space = 282
            # critic uses privileged state (same as baseline's 55D) — no override needed
        self.cfg_task = cfg.task

        # --- Sensor attributes (must be set BEFORE super().__init__ because
        # DirectRLEnv.__init__ calls _setup_scene which references them) ---
        self._tiled_camera_left: TiledCamera | None = None
        self._tiled_camera_right: TiledCamera | None = None
        self.hand_body_idx: int | None = None

        super().__init__(cfg, render_mode, **kwargs)

        # Apply render quality settings BEFORE first physics step so the
        # renderer picks them up. RasterizedLighting + no RTX gives ~5-10x
        # faster camera renders with negligible quality loss at 224×224.
        self._apply_render_quality()

        self._set_body_inertias()
        self._init_tensors()
        self._set_default_dynamics_parameters()

        # --- Hand body index + FT buffers (must be before _compute_intermediate_values) ---
        self.hand_body_idx = self._robot.body_names.index("panda_hand")

        self.ft_history_len = FT_FILTER_CFG["history_len"]
        self.ft_history = torch.zeros((self.num_envs, self.ft_history_len, 6), device=self.device)
        self.ft_history_ptr = 0
        self.force_torque = torch.zeros((self.num_envs, 6), device=self.device)
        self.delta_f = torch.zeros((self.num_envs, 6), device=self.device)
        self.contact_force_state = torch.zeros((self.num_envs, 4), device=self.device)
        self.engagement_state = torch.zeros((self.num_envs, 1), device=self.device)

        # Raw force ring buffer: K+10 frames
        # ft_ring[-10:] = current 10-frame, ft_ring[:10] = K steps ago
        self.K = 3
        self.ft_ring = torch.zeros((self.num_envs, self.K + 10, 6), device=self.device)

        self._compute_intermediate_values(dt=self.physics_dt)

        # Init camera attributes EARLY (before DR) with same defaults as _update_camera_poses
        self._eye1_local = [-0.08, -0.0, 0.08]
        self._eye2_local = [0.04, 0.0, 0.02]
        self._target_left_local = [0.0, 0.0, 0.17]
        self._target_right_local = [0.0, 0.0, 0.17]

        # Load pretrained encoder if checkpoint path is configured.
        # Skip in debug mode — the encoder is never called (policy = critic = state).
        from .encoder import VisualForceEncoder
        encoder_ckpt = getattr(cfg, "encoder_checkpoint", None)
        self._encoder: VisualForceEncoder | None = None
        if encoder_ckpt is not None and encoder_ckpt != "" \
                and not self._encoder_debug_state_policy \
                and not self._encoder_ablate_bottleneck:
            backbone_name = getattr(cfg, "encoder_backbone", "resnet18")
            mono = getattr(cfg, "encoder_mono", False)
            self._encoder = VisualForceEncoder.from_checkpoint(
                str(encoder_ckpt), backbone_name=backbone_name, device=self.device,
                mono=mono)
            print(f"[INFO] Loaded frozen encoder from {encoder_ckpt} "
                  f"(feature_dim={self._encoder.feature_dim})")

    def _apply_render_quality(self):
        """Apply carb render-quality settings from the env config.

        Called once during ``__init__``, before the first render step.
        ``RasterizedLighting`` + all RTX off → ~5-10x faster camera renders
        at the cost of ray-traced shadows/GI (negligible at 224×224 for RL).
        """
        try:
            import carb
            s = carb.settings.get_settings()
        except Exception:
            return

        render_mode = getattr(self.cfg, "render_mode", "")
        if render_mode:
            try:
                s.set("/rtx/rendermode", str(render_mode))
            except Exception:
                pass

        bool_flags = [
            ("rtx_ao", "/rtx/ambientOcclusion/enabled"),
            ("rtx_shadows", "/rtx/shadows/enabled"),
            ("rtx_reflections", "/rtx/reflections/enabled"),
            ("rtx_gi", "/rtx/indirectDiffuse/enabled"),
        ]
        for cfg_key, carb_path in bool_flags:
            val = getattr(self.cfg, cfg_key, None)
            if val is not None:
                try:
                    s.set(carb_path, bool(val))
                except Exception:
                    pass

        spp = getattr(self.cfg, "rtx_spp", None)
        if spp is not None:
            try:
                s.set("/rtx/pathtracing/spp", int(spp))
            except Exception:
                pass

        # Disable post-processing — unnecessary for RL (encoder sees patches, not pixels)
        try:
            s.set("/rtx/post/aa/op", 0)        # anti-aliasing off
            # s.set("/rtx/post/tonemap/op", 0)  # TONE MAPPING KEPT ON — fix white images
        except Exception:
            pass

    def _set_body_inertias(self):
        """Note: this is to account for the asset_options.armature parameter in IGE."""
        inertias = self._robot.root_physx_view.get_inertias()
        offset = torch.zeros_like(inertias)
        offset[:, :, [0, 4, 8]] += 0.01
        new_inertias = inertias + offset
        self._robot.root_physx_view.set_inertias(new_inertias, torch.arange(self.num_envs))

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Set masses and frictions.
        self._set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction)
        self._set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction)
        self._set_friction(self._robot, self.cfg_task.robot_cfg.friction)

    def _set_friction(self, asset, value):
        """Update material properties for a given asset."""
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(self.scene.num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def _init_tensors(self):
        """Initialize tensors once."""
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)

        # Fixed asset.
        self.fixed_pos_action_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Held asset
        held_base_x_offset = 0.0
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "usb_insert":
            held_base_z_offset = 0.0
        elif self.cfg_task.name == "gear_mesh":
            gear_base_offset = self._get_target_gear_base_offset()
            held_base_x_offset = gear_base_offset[0]
            held_base_z_offset = gear_base_offset[2]
        elif self.cfg_task.name == "nut_thread":
            held_base_z_offset = self.cfg_task.fixed_asset_cfg.base_height
        else:
            raise NotImplementedError("Task not implemented")

        self.held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.held_base_pos_local[:, 0] = held_base_x_offset
        self.held_base_pos_local[:, 2] = held_base_z_offset
        self.held_base_quat_local = self.identity_quat.clone().detach()

        self.held_base_pos = torch.zeros_like(self.held_base_pos_local)
        self.held_base_quat = self.identity_quat.clone().detach()

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = self.identity_quat.clone()
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        # Keypoint tensors.
        self.target_held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_held_base_quat = self.identity_quat.clone().detach()

        offsets = self._get_keypoint_offsets(self.cfg_task.num_keypoints)
        self.keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        self.keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        self.keypoints_fixed = torch.zeros_like(self.keypoints_held, device=self.device)

        # Used to compute target poses.
        self.fixed_success_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "usb_insert":
            self.fixed_success_pos_local[:, 2] = 0.0
        elif self.cfg_task.name == "gear_mesh":
            gear_base_offset = self._get_target_gear_base_offset()
            self.fixed_success_pos_local[:, 0] = gear_base_offset[0]
            self.fixed_success_pos_local[:, 2] = gear_base_offset[2]
        elif self.cfg_task.name == "nut_thread":
            head_height = self.cfg_task.fixed_asset_cfg.base_height
            shank_length = self.cfg_task.fixed_asset_cfg.height
            thread_pitch = self.cfg_task.fixed_asset_cfg.thread_pitch
            self.fixed_success_pos_local[:, 2] = head_height + shank_length - thread_pitch * 1.5
        else:
            raise NotImplementedError("Task not implemented")

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.4))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        # --- Create cameras BEFORE clone (so prims exist in template env_0) ---
        if self.cfg.tiled_camera_left is not None:
            self._tiled_camera_left = TiledCamera(self.cfg.tiled_camera_left)
        if self.cfg.tiled_camera_right is not None:
            self._tiled_camera_right = TiledCamera(self.cfg.tiled_camera_right)

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        # --- Register cameras with scene after clone ---
        if self._tiled_camera_left is not None:
            self.scene.sensors["tiled_camera_left"] = self._tiled_camera_left
        if self._tiled_camera_right is not None:
            self.scene.sensors["tiled_camera_right"] = self._tiled_camera_right

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        # Keypoint tensors.
        self.held_base_quat[:], self.held_base_pos[:] = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.held_base_quat_local, self.held_base_pos_local
        )
        self.target_held_base_quat[:], self.target_held_base_pos[:] = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, self.fixed_success_pos_local
        )

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_held[:, idx] = torch_utils.tf_combine(
                self.held_base_quat, self.held_base_pos, self.identity_quat, keypoint_offset.repeat(self.num_envs, 1)
            )[1]
            self.keypoints_fixed[:, idx] = torch_utils.tf_combine(
                self.target_held_base_quat,
                self.target_held_base_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

        self.keypoint_dist = torch.norm(self.keypoints_held - self.keypoints_fixed, p=2, dim=-1).mean(-1)
        self.last_update_timestamp = self._robot._data._sim_timestamp

        # Update FT (camera poses updated once per step in _pre_physics_step
        self._update_force_torque()

    def _update_force_torque(self):
        """Read net contact force on the peg via fingertip force summation.

        Minimal pipeline (per paper consensus — MCR/AFRO/MSDP):
          raw fingertip forces → sum left+right → normalize(/50.0)

        k=3 differencing in the pretraining architecture (idea.md §3) naturally
        cancels the DC offset from gravity + gripper holding force.  No EMA,
        baseline subtraction, deadzone, or clipping — the network learns its
        own representations from the full dynamic range.
        """
        if not hasattr(self, "left_finger_body_idx"):
            return

        raw_wrench = self._robot.root_physx_view.get_link_incoming_joint_force()
        left_f  = raw_wrench[:, self.left_finger_body_idx, :]
        right_f = raw_wrench[:, self.right_finger_body_idx, :]
        raw_ft  = left_f + right_f  # (N, 6)

        # No hand-coded normalisation — ForceMLP's LayerNorm learns the
        # optimal scaling from data (sim + real).
        self.force_torque[:, :] = raw_ft

        # Push to ring buffer
        self.ft_ring[:, :-1, :] = self.ft_ring[:, 1:, :].clone()
        self.ft_ring[:, -1, :] = raw_ft

        # 5-frame ring buffer (for delta_f computation)
        self.ft_history[:, self.ft_history_ptr, :] = self.force_torque
        self.ft_history_ptr = (self.ft_history_ptr + 1) % self.ft_history_len

        prev_ptr = (self.ft_history_ptr - 3) % self.ft_history_len
        self.delta_f[:, :] = self.force_torque - self.ft_history[:, prev_ptr, :]

        # --- 4-class contact-force state (aligned with idea.md §B) ---
        # Defined purely by structural force features — cross-domain stable:
        #   FREE (1,0,0,0):         |F|_max < τ  → no contact
        #   CONTACT_UP (0,1,0,0):   |F|_max ≥ τ AND max(ΔF) > ε  → force rising
        #   CONTACT_DOWN (0,0,1,0): |F|_max ≥ τ AND min(ΔF) < −ε → force dropping
        #   CONTACT_STEADY (0,0,0,1):|F|_max ≥ τ AND |ΔF|_max ≤ ε → contact, stable
        # τ (noise floor): deadzone_n / clip_n in normalized space
        # ε (force delta threshold): 0.02 normalized (≈ 1.0N raw)
        # ΔF = F_t - F_{t-3}  (k=3 step force difference, aligned with idea.md)
        tau = FT_FILTER_CFG["deadzone_n"] / FT_FILTER_CFG["clip_n"]  # ≈ 0.02
        eps = 0.02  # force delta threshold in normalized space (~1.0N raw)

        f_mag = self.force_torque.abs().max(dim=-1).values  # (N,) max abs across 6 axes
        df_mag = self.delta_f.abs().max(dim=-1).values               # (N,) max abs ΔF

        is_free = f_mag < tau
        df_rising = self.delta_f.max(dim=-1).values > eps
        df_falling = self.delta_f.min(dim=-1).values < -eps
        in_contact = ~is_free

        self.contact_force_state[:, 0] = is_free.float()
        self.contact_force_state[:, 1] = (in_contact & df_rising).float()
        self.contact_force_state[:, 2] = (in_contact & df_falling).float()
        self.contact_force_state[:, 3] = (in_contact & ~df_rising & ~df_falling).float()

        # Engagement state: inherited from prior keypoint-based definition.
        engage_dist = self.cfg_task.engage_threshold * self.cfg_task.fixed_asset_cfg.height
        self.engagement_state[:, 0] = (self.keypoint_dist <= engage_dist).float()

    def _update_camera_poses(self):
        """Update camera world poses to track the panda_hand body.

        Uses ``set_world_poses_from_view`` to avoid MuJoCo→IsaacLab quaternion
        convention issues. Both cameras look at the peg tip area below the hand.

        Body-local offsets (mujoco panda.xml hand body frame, X-fwd Y-left Z-up):
          cam1 (left):  eye=(-0.10, 0, 0.07) — behind hand center, above
          cam2 (right): eye=( 0.10, 0, 0.07)
          target:       ( 0.05, 0, -0.12) — peg tip, forward and below hand center
        """
        if self._tiled_camera_left is None:
            return

        hand_pos = self._robot.data.body_pos_w[:, self.hand_body_idx]  # (N, 3)
        hand_quat = self._robot.data.body_quat_w[:, self.hand_body_idx]  # (N, 4) wxyz

        # 这样就可以不用折腾四元数了，只用规定相机的注视点
        # Body-local offsets — separate targets for each camera
        e1 = self._eye1_local; e2 = self._eye2_local
        t1 = self._target_left_local; t2 = self._target_right_local
        eye1_local = torch.tensor(e1, device=self.device)
        eye2_local = torch.tensor(e2, device=self.device)
        tgt1_local = torch.tensor(t1, device=self.device)
        tgt2_local = torch.tensor(t2, device=self.device)

        # Transform to world frame
        eye1_world = hand_pos + quat_apply(hand_quat, eye1_local.expand_as(hand_pos))
        eye2_world = hand_pos + quat_apply(hand_quat, eye2_local.expand_as(hand_pos))
        tgt1_world = hand_pos + quat_apply(hand_quat, tgt1_local.expand_as(hand_pos))
        tgt2_world = hand_pos + quat_apply(hand_quat, tgt2_local.expand_as(hand_pos))

        self._tiled_camera_left.set_world_poses_from_view(eye1_world, tgt1_world)
        self._tiled_camera_right.set_world_poses_from_view(eye2_world, tgt2_world)

    def _compute_encoder_features(self):
        """Return (z_vis(128D), z_force(128D)) from pretrained encoder."""
        if self._encoder is None or self._tiled_camera_left is None:
            return None
        from .encoder import preprocess_rgb
        img_l = preprocess_rgb(self._tiled_camera_left.data.output["rgb"])
        img_r = None
        if self._tiled_camera_right is not None:
            img_r = preprocess_rgb(self._tiled_camera_right.data.output["rgb"])

        ft_3f   = self.ft_ring[:, -10:, :].reshape(self.num_envs, 60)   # 10-frame
        ft_3f_k = self.ft_ring[:, :10, :].reshape(self.num_envs, 60)    # K steps ago
        proprio_20 = torch.cat([
            self.joint_pos[:, 0:7], self.fingertip_midpoint_pos,
            self.fingertip_midpoint_quat, self.ee_linvel_fd, self.ee_angvel_fd,
        ], dim=-1)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            z_vis, z_force, _, _ = self._encoder.forward_features(
                img_l, img_r, ft_3f, ft_3f_k, self.raw_actions, proprio_20)
        return z_vis.float(), z_force.float()  # (N, 128) each

    def _get_observations(self):
        """Get actor/critic inputs.

        With encoder: policy = [Δv(vis_dim), Δf(64), proprio(20D), prev_action(6D)].
        Without:      policy = 25D low-dim (proprio + force + prev_actions).
        """
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "force_torque": self.force_torque,
            "contact_force_state": self.contact_force_state,
            "engagement_state": self.engagement_state,
            "keypoint_dist": self.keypoint_dist.unsqueeze(-1),
            "fingertip_dir": F.normalize(
                self.fingertip_midpoint_pos - noisy_fixed_pos, dim=-1, eps=1e-8),
            "fingertip_dir_xy": F.normalize(
                (self.fingertip_midpoint_pos - noisy_fixed_pos)[:, :2], dim=-1, eps=1e-8),
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "force_torque": self.force_torque,
            "contact_force_state": self.contact_force_state,
            "engagement_state": self.engagement_state,
            "keypoint_dist": self.keypoint_dist.unsqueeze(-1),
            "prev_actions": prev_actions,
        }

        # ── Encoder-based observations (Stage F) ──
        if self._encoder_debug_state_policy:
            # Debug mode: actor sees same privileged state as critic.
            critic_obs = torch.cat([state_dict[name] for name in self.cfg.state_order + ["prev_actions"]], dim=-1)
            observations = {"policy": critic_obs, "critic": critic_obs}
        elif self._encoder_ablate_bottleneck:
            # Ablation: held_pos_rel_fixed from privileged state, no encoder.
            # Policy = [held_pos_rel_fixed(3), proprio(20), prev_a(6)] = 29D
            held_rel = self.held_pos - self.fixed_pos_obs_frame  # peg→hole 3D
            proprio_20 = torch.cat([
                self.joint_pos[:, 0:7],
                self.fingertip_midpoint_pos,
                self.fingertip_midpoint_quat,
                self.ee_linvel_fd,
                self.ee_angvel_fd,
            ], dim=-1)
            policy_obs = torch.cat([held_rel, proprio_20, prev_actions], dim=-1)
            critic_obs = torch.cat([state_dict[name] for name in self.cfg.state_order + ["prev_actions"]], dim=-1)
            observations = {"policy": policy_obs, "critic": critic_obs}
        else:
            enc_out = self._compute_encoder_features()
            if enc_out is not None:
                z_vis, z_force = enc_out
                proprio_20 = torch.cat([
                    self.joint_pos[:, 0:7],
                    self.fingertip_midpoint_pos,
                    self.fingertip_midpoint_quat,
                    self.ee_linvel_fd,
                    self.ee_angvel_fd,
                ], dim=-1)
                # Policy = [z_vis(128), z_force(128), proprio(20), prev_a(6)] = 282D
                policy_obs = torch.cat([z_vis, z_force, proprio_20, prev_actions], dim=-1)
                critic_obs = torch.cat(
                    [state_dict[name] for name in self.cfg.state_order + ["prev_actions"]], dim=-1)
                observations = {"policy": policy_obs, "critic": critic_obs}
            else:
                # ── Low-dim observations (backward-compatible) ──
                obs_tensors = [obs_dict[name] for name in self.cfg.obs_order + ["prev_actions"]]
                obs_tensors = torch.cat(obs_tensors, dim=-1)
                state_tensors = [state_dict[name] for name in self.cfg.state_order + ["prev_actions"]]
                state_tensors = torch.cat(state_tensors, dim=-1)
                observations = {"policy": obs_tensors, "critic": state_tensors}

        if self._tiled_camera_left is not None:
            observations["camera_left_rgb"] = self._tiled_camera_left.data.output["rgb"]
        if self._tiled_camera_right is not None:
            observations["camera_right_rgb"] = self._tiled_camera_right.data.output["rgb"]

        return observations

    def get_camera_data(self) -> dict | None:
        """Return camera images for external consumers (Stage D/E encoder pretraining).

        Returns:
            Dict with keys ``"left_rgb"`` and ``"right_rgb"``, each a tensor of
            shape ``(num_envs, height, width, 3)`` in [0, 255] uint8 range.
            Returns ``None`` if cameras are not configured.
        """
        if self._tiled_camera_left is None or self._tiled_camera_right is None:
            return None
        return {
            "left_rgb": self._tiled_camera_left.data.output["rgb"],
            "right_rgb": self._tiled_camera_right.data.output["rgb"],
        }

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0
        # Clear force ring buffer for reset envs
        self.ft_ring[env_ids] = 0.0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.raw_actions = action.clone().to(self.device)  # before EMA
        self.actions = (
            self.cfg.ctrl.ema_factor * self.raw_actions + (1 - self.cfg.ctrl.ema_factor) * self.actions
        )

        # Update camera poses BEFORE physics rendering so cameras track the hand
        # for the current step. (Also called in _compute_intermediate_values for
        # the FT pipeline; the duplicate is cheap.)
        if self._tiled_camera_left is not None:
            self._update_camera_poses()

    def close_gripper_in_place(self):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)
        ctrl_target_gripper_dof_pos = 0.0

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.generate_ctrl_signals()

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Get current yaw for success checking.
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        self.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold

        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        delta_pos = self.ctrl_target_fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        self.ctrl_target_fingertip_midpoint_pos = self.fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = 0.0
        self.generate_ctrl_signals()

    def _set_gains(self, prop_gains, rot_deriv_scale=1.0):
        """Set robot gains using critical damping."""
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = 2 * torch.sqrt(prop_gains)
        self.task_deriv_gains[:, 3:6] /= rot_deriv_scale

    def generate_ctrl_signals(self):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = fc.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,  # _fd,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.ee_linvel_fd,
            fingertip_midpoint_angvel=self.ee_angvel_fd,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
        )

        # set target for gripper joints to use physx's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = self.ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

        # Update camera poses each decimation step so rendering captures the current
        # hand position. (Also called from _pre_physics_step for the first frame.)
        if self._tiled_camera_left is not None:
            self._update_camera_poses()

    def _get_dones(self):
        """Update intermediate values used for rewards and observations."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        xy_dist = torch.linalg.vector_norm(self.target_held_base_pos[:, 0:2] - self.held_base_pos[:, 0:2], dim=1)
        z_disp = self.held_base_pos[:, 2] - self.target_held_base_pos[:, 2]

        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
        # Height threshold to target
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name in ("peg_insert", "gear_mesh", "usb_insert"):
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold
        else:
            raise NotImplementedError("Task not implemented")
        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        if check_rot:
            is_rotated = self.curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)

        return curr_successes

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_buf = self._update_rew_buf(curr_successes)

        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        # Get the time at which an episode first succeeds.
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:  # Only log for successful episodes.
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        self.prev_actions = self.actions.clone()
        # Store current success for curriculum (consumed by _get_observations next call)
        self._curriculum_curr_success_rate = curr_successes.float().mean().item()
        return rew_buf

    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep."""
        rew_dict = {}

        # Keypoint rewards.
        def squashing_fn(x, a, b):
            return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))

        a0, b0 = self.cfg_task.keypoint_coef_baseline
        rew_dict["kp_baseline"] = squashing_fn(self.keypoint_dist, a0, b0)
        # a1, b1 = 25, 2
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        rew_dict["kp_coarse"] = squashing_fn(self.keypoint_dist, a1, b1)
        a2, b2 = self.cfg_task.keypoint_coef_fine
        # a2, b2 = 300, 0
        rew_dict["kp_fine"] = squashing_fn(self.keypoint_dist, a2, b2)

        # Action penalties.
        rew_dict["action_penalty"] = torch.norm(self.actions, p=2)
        rew_dict["action_grad_penalty"] = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        rew_dict["curr_engaged"] = (
            self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False).clone().float()
        )
        rew_dict["curr_successes"] = curr_successes.clone().float()

        rew_buf = (
            rew_dict["kp_coarse"]
            + rew_dict["kp_baseline"]
            + rew_dict["kp_fine"]
            - rew_dict["action_penalty"] * self.cfg_task.action_penalty_scale
            - rew_dict["action_grad_penalty"] * self.cfg_task.action_grad_penalty_scale
            + rew_dict["curr_engaged"]
            + rew_dict["curr_successes"]
        )

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        return rew_buf

    def _reset_idx(self, env_ids):
        """
        We assume all envs will always be reset at the same time.
        """
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        # Material replacement BEFORE physics step (avoids setGlobalPose GPU error)
        self._randomize_materials(self.cfg.domain_rand, env_ids)
        self.step_sim_no_action()

        # Per-episode domain randomization (lighting, physics)
        self._apply_domain_randomization(env_ids)

        self.randomize_initial_state(env_ids)

        # Reset FT state for reset envs.
        self.ft_history[env_ids] = 0.0
        self.ft_ring[env_ids] = 0.0
        self.force_torque[env_ids] = 0.0
        self.delta_f[env_ids] = 0.0
        self.contact_force_state[env_ids] = 0.0
        self.engagement_state[env_ids] = 0.0

    def _get_target_gear_base_offset(self):
        """Get offset of target gear from the gear base asset."""
        target_gear = self.cfg_task.target_gear
        if target_gear == "gear_large":
            gear_base_offset = self.cfg_task.fixed_asset_cfg.large_gear_base_offset
        elif target_gear == "gear_medium":
            gear_base_offset = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset
        elif target_gear == "gear_small":
            gear_base_offset = self.cfg_task.fixed_asset_cfg.small_gear_base_offset
        else:
            raise ValueError(f"{target_gear} not valid in this context!")
        return gear_base_offset

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def set_pos_inverse_kinematics(self, env_ids):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            # Compute error to target.
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """Get default relative pose between help asset and fingertip."""
        if self.cfg_task.name in ("peg_insert", "usb_insert"):
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
        elif self.cfg_task.name == "gear_mesh":
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            gear_base_offset = self._get_target_gear_base_offset()
            held_asset_relative_pos[:, 0] += gear_base_offset[0]
            held_asset_relative_pos[:, 2] += gear_base_offset[2]
            held_asset_relative_pos[:, 2] += self.cfg_task.held_asset_cfg.height / 2.0 * 1.1
        elif self.cfg_task.name == "nut_thread":
            held_asset_relative_pos = self.held_base_pos_local
        else:
            raise NotImplementedError("Task not implemented")

        held_asset_relative_quat = self.identity_quat
        if self.cfg_task.name == "nut_thread":
            # Rotate along z-axis of frame for default position.
            initial_rot_deg = self.cfg_task.held_asset_rot_init
            rot_yaw_euler = torch.tensor([0.0, 0.0, initial_rot_deg * np.pi / 180.0], device=self.device).repeat(
                self.num_envs, 1
            )
            held_asset_relative_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_yaw_euler[:, 0], pitch=rot_yaw_euler[:, 1], yaw=rot_yaw_euler[:, 2]
            )

        return held_asset_relative_pos, held_asset_relative_quat

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def _apply_domain_randomization(self, env_ids):
        """Apply per-episode visual, lighting, and physics randomization.

        Draws parameters from ``self.cfg.domain_rand`` ranges.  Called once per
        reset so every episode sees a different randomised configuration.
        """
        dr = self.cfg.domain_rand
        n = len(env_ids)

        def _uniform(lo: float, hi: float) -> float:
            return float(lo + (hi - lo) * torch.rand(1).item())

        # ── Lighting randomization (carb settings — affects all envs) ──
        try:
            import carb
            s = carb.settings.get_settings()
        except Exception:
            s = None

        if s is not None:
            if dr.dome_light_intensity[1] > dr.dome_light_intensity[0]:
                s.set("/rtx/domeLight/intensity",
                      _uniform(*dr.dome_light_intensity) * 120.0)
            if dr.key_light_intensity[1] > dr.key_light_intensity[0]:
                s.set("/rtx/distantLight/intensity",
                      _uniform(*dr.key_light_intensity) * 450.0)

        # ── Camera position noise (modify eye_local) ──
        if self._tiled_camera_left is not None:
            # Read orig from current values (set by _update_camera_poses defaults)
            if not hasattr(self, "_eye1_local_orig"):
                # Copy from current values (set in __init__)
                self._eye1_local_orig = list(self._eye1_local)
                self._eye2_local_orig = list(self._eye2_local)
                self._target_left_local_orig = list(self._target_left_local)
                self._target_right_local_orig = list(self._target_right_local)

            cp = dr.camera_pos_noise
            ct = dr.camera_target_noise
            self._eye1_local = [
                self._eye1_local_orig[0] + _uniform(-cp[0], cp[0]),
                self._eye1_local_orig[1] + _uniform(-cp[1], cp[1]),
                self._eye1_local_orig[2] + _uniform(-cp[2], cp[2]),
            ]
            self._eye2_local = [
                self._eye2_local_orig[0] + _uniform(-cp[0], cp[0]),
                self._eye2_local_orig[1] + _uniform(-cp[1], cp[1]),
                self._eye2_local_orig[2] + _uniform(-cp[2], cp[2]),
            ]
            self._target_left_local = [
                self._target_left_local_orig[0] + _uniform(-ct[0], ct[0]),
                self._target_left_local_orig[1] + _uniform(-ct[1], ct[1]),
                self._target_left_local_orig[2] + _uniform(-ct[2], ct[2]),
            ]
            self._target_right_local = [
                self._target_right_local_orig[0] + _uniform(-ct[0], ct[0]),
                self._target_right_local_orig[1] + _uniform(-ct[1], ct[1]),
                self._target_right_local_orig[2] + _uniform(-ct[2], ct[2]),
            ]

        # ── Physics: mass randomization ──
        if dr.held_mass_scale[1] > dr.held_mass_scale[0]:
            scale = _uniform(*dr.held_mass_scale)
            try:
                mass_api = self._held_asset.root_physx_view.get_masses()
                if mass_api is not None:
                    default_mass = self._held_asset.default_mass.clone() if hasattr(self._held_asset, 'default_mass') else mass_api[env_ids].clone()
                    new_mass = default_mass * scale
                    self._held_asset.root_physx_view.set_masses(new_mass, env_ids)
            except Exception:
                pass

        # ── Physics: friction randomization ──
        if dr.held_friction[1] > dr.held_friction[0]:
            sf = _uniform(*dr.held_friction)
            df = sf * 0.85
            try:
                mats = self._held_asset.root_physx_view.get_material_properties()
                mats[env_ids, 0] = sf   # static
                mats[env_ids, 1] = df   # dynamic
                self._held_asset.root_physx_view.set_material_properties(mats, env_ids)
            except Exception:
                pass

        # ── Physics: robot dynamics randomization ──
        if dr.joint_damping_scale[1] > dr.joint_damping_scale[0]:
            kd_scale = _uniform(*dr.joint_damping_scale)
            kp_scale = _uniform(*dr.joint_stiffness_scale) if dr.joint_stiffness_scale[1] > dr.joint_stiffness_scale[0] else 1.0
            try:
                gains = self._robot.root_physx_view.get_dof_stiffnesses_and_dampings()
                if gains is not None:
                    kp, kd = gains
                    kp_new = kp.clone()
                    kd_new = kd.clone()
                    for j in range(min(7, kd_new.shape[1])):
                        kp_new[env_ids, j] *= kp_scale
                        kd_new[env_ids, j] *= kd_scale
                    self._robot.root_physx_view.set_dof_stiffnesses_and_dampings(kp_new, kd_new, env_ids)
            except Exception:
                pass

        # ── Physics: peg/hole scale randomization ──
        if dr.peg_scale[1] > dr.peg_scale[0]:
            peg_s = _uniform(*dr.peg_scale)
            try:
                self._held_asset.root_physx_view.set_scales(
                    torch.full((n, 3), peg_s, device=self.device), env_ids)
            except Exception:
                pass




    # ── Material presets for full replacement ──
    MATERIAL_PRESETS = [
        # (diffuse RGB, roughness, metallic, opacity)
        # Matte plastics — saturated
        ((0.90, 0.05, 0.05), 0.5, 0.0, 1.0),   # 0: bright red
        ((0.05, 0.15, 0.90), 0.4, 0.0, 1.0),   # 1: deep blue
        ((0.05, 0.80, 0.10), 0.5, 0.0, 1.0),   # 2: vivid green
        ((0.95, 0.90, 0.05), 0.3, 0.0, 1.0),   # 3: bright yellow
        ((0.810, 0.17, 0.07), 0.4, 0.5, 1.0),   # 4: coral #e8734a
        ((0.70, 0.05, 0.60), 0.4, 0.0, 1.0),   # 5: magenta
        ((0.05, 0.70, 0.70), 0.4, 0.0, 1.0),   # 6: cyan
        ((0.98, 0.98, 0.98), 0.3, 0.0, 1.0),   # 7: white matte
        ((0.05, 0.05, 0.08), 0.6, 0.0, 1.0),   # 8: near-black matte
        # Metals
        ((0.90, 0.90, 0.92), 0.1, 0.9, 1.0),   # 9: polished steel
        ((0.85, 0.60, 0.20), 0.2, 0.7, 1.0),   # 10: brass/gold
        ((0.55, 0.35, 0.20), 0.3, 0.5, 1.0),   # 11: bronze
        ((0.20, 0.20, 0.25), 0.5, 0.9, 1.0),   # 12: dark gunmetal
        # Rubber/soft
        ((0.25, 0.22, 0.22), 0.85, 0.05, 1.0), # 13: dark brown rubber
        ((0.40, 0.40, 0.38), 0.75, 0.1, 1.0),  # 14: grey rubber
        ((0.08, 0.35, 0.08), 0.7, 0.05, 1.0),  # 15: dark green rubber
        # Extremes for table
        ((0.98, 0.96, 0.88), 0.2, 0.0, 1.0),   # 16: cream/off-white
        ((0.40, 0.25, 0.10), 0.6, 0.0, 1.0),   # 17: brown wood
        ((0.55, 0.50, 0.55), 0.3, 0.4, 1.0),   # 18: slate grey metallic
        ((0.82, 0.75, 0.65), 0.5, 0.0, 1.0),   # 19: light wood/beige
        # High-contrast neon
        ((0.10, 0.95, 0.20), 0.2, 0.0, 1.0),   # 20: neon green
        ((1.00, 0.10, 0.60), 0.3, 0.0, 1.0),   # 21: hot pink
        ((0.15, 0.80, 0.95), 0.2, 0.0, 1.0),   # 22: sky blue
    ]

    def _replace_material(self, prim, preset_idx: int):
        """Create new material/shaders and bind to all descendant prims."""
        try:
            from pxr import UsdShade, Sdf, Gf
            stage = prim.GetStage()
            color, roughness, metallic, opacity = self.MATERIAL_PRESETS[preset_idx]

            mat_path = prim.GetPath().AppendChild(f"material_preset_{preset_idx}")
            mat = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("shader"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

            def _bind_all(p):
                try:
                    UsdShade.MaterialBindingAPI(p).Bind(mat)
                except Exception:
                    pass
                for child in p.GetAllChildren():
                    _bind_all(child)
            _bind_all(prim)
        except Exception:
            pass

    def _randomize_materials(self, dr, env_ids):
        """Randomize material appearance: param tweaks OR full preset replacement."""
        def _rand(lo: float, hi: float) -> float:
            return float(lo + (hi - lo) * torch.rand(1).item())

        try:
            import omni.usd
            from pxr import UsdShade, Gf
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return
        except Exception:
            return

        # ── Hole material: preset 4 (neon orange) ──
        if getattr(dr, "material_full_replace", False):
            try:
                fixed_prim = stage.GetPrimAtPath(
                    self._fixed_asset.cfg.prim_path.replace("/World/envs/env_.*/", "/World/envs/env_0/"))
                if fixed_prim and fixed_prim.IsValid():
                    self._replace_material(fixed_prim, 4)
            except Exception:
                pass
            return



        def _hsv_shift(rgb_in: tuple, hue_delta: float, sat_mul: float, val_mul: float) -> tuple:
            """Shift hue (0-360°), multiply saturation and value."""
            import colorsys
            r, g, b = [max(0, min(1, float(c))) for c in rgb_in]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h = (h * 360.0 + hue_delta) % 360.0 / 360.0
            s = max(0, min(1, s * sat_mul))
            v = max(0, min(1, v * val_mul))
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            return (r, g, b)

        def _modify_shader(shader, *, color_delta=None, roughness=None, metallic=None,
                           color_shift=None, hue_shift=0, sat_mul=1.0, val_mul=1.0):
            """Modify common UsdPreviewSurface shader inputs."""
            if shader is None:
                return
            # Diffuse color
            if color_delta or hue_shift or sat_mul != 1.0 or val_mul != 1.0:
                inp = shader.GetInput("diffuseColor")
                if inp:
                    try:
                        val = inp.Get()
                        if val is not None:
                            rgb = tuple(float(val[i]) for i in range(3))
                            if hue_shift or sat_mul != 1.0 or val_mul != 1.0:
                                rgb = _hsv_shift(rgb, hue_shift, sat_mul, val_mul)
                            if color_delta:
                                rgb = tuple(max(0, min(1, rgb[i] + color_delta[i])) for i in range(3))
                            inp.Set(Gf.Vec3f(*rgb))
                    except Exception:
                        pass
            if roughness is not None:
                inp = shader.GetInput("roughness")
                if inp:
                    try:
                        inp.Set(float(roughness))
                    except Exception:
                        pass
            if metallic is not None:
                inp = shader.GetInput("metallic")
                if inp:
                    try:
                        inp.Set(float(metallic))
                    except Exception:
                        pass

        def _randomize_prim_materials(prim, **kwargs):
            """Walk a prim and randomize all bound materials."""
            if prim is None:
                return
            try:
                from pxr import UsdShade
                for p in [prim] + list(prim.GetAllChildren()):
                    mat_binding = UsdShade.MaterialBindingAPI(p)
                    try:
                        mat, _ = mat_binding.ComputeBoundMaterial()
                    except Exception:
                        continue
                    if mat is None:
                        continue
                    surface = mat.GetSurface() if hasattr(mat, 'GetSurface') else None
                    _modify_shader(surface, **kwargs)
                    # Also try volume / displacement for full coverage
                    for api_name in ('GetVolume', 'GetDisplacement'):
                        api = getattr(mat, api_name, None)
                        if callable(api):
                            try:
                                s = api()
                                _modify_shader(s, **kwargs)
                            except Exception:
                                pass
            except Exception:
                pass

        # ── Held asset (peg): random hue / saturation / brightness ──
        if dr.peg_hue_shift[1] > dr.peg_hue_shift[0] or dr.peg_saturation[1] > 1.0 or dr.peg_value[1] > 1.0:
            hue = _rand(dr.peg_hue_shift[0], dr.peg_hue_shift[1]) if dr.peg_hue_shift[1] > dr.peg_hue_shift[0] else 0
            sat = _rand(dr.peg_saturation[0], dr.peg_saturation[1]) if dr.peg_saturation[1] > 1.0 else 1.0
            val = _rand(dr.peg_value[0], dr.peg_value[1]) if dr.peg_value[1] > 1.0 else 1.0
            try:
                prim = self._held_asset.prim if hasattr(self._held_asset, 'prim') else None
                if prim is not None:
                    _randomize_prim_materials(prim, hue_shift=hue, sat_mul=sat, val_mul=val)
            except Exception:
                pass

        # ── Table: random roughness / metallic / color shift ──
        if dr.table_roughness[1] > dr.table_roughness[0] or dr.table_metallic[1] > dr.table_metallic[0] or dr.table_color_shift[1] > 0:
            try:
                table_prim_path = "/World/envs/env_0/Table"
                prim = stage.GetPrimAtPath(table_prim_path)
                if prim:
                    roughness = _rand(*dr.table_roughness) if dr.table_roughness[1] > dr.table_roughness[0] else None
                    metallic = _rand(*dr.table_metallic) if dr.table_metallic[1] > dr.table_metallic[0] else None
                    shift = dr.table_color_shift[1]
                    color_delta = (_rand(-shift, shift), _rand(-shift, shift), _rand(-shift, shift)) if shift > 0 else None
                    _randomize_prim_materials(prim, roughness=roughness, metallic=metallic, color_delta=color_delta)
            except Exception:
                pass

        # ── Floor / room: random hue ──
        if dr.floor_hue_shift[1] > dr.floor_hue_shift[0]:
            hue = _rand(*dr.floor_hue_shift)
            try:
                for path in ("/World/MonteTask/floor_visual", "/World/envs/env_0/floor_visual"):
                    prim = stage.GetPrimAtPath(path)
                    if prim:
                        _randomize_prim_materials(prim, hue_shift=hue, sat_mul=0.8, val_mul=1.0)
            except Exception:
                pass

        # ── Fixed asset (hole fixture): random hue ──
        if dr.fixture_hue_shift[1] > dr.fixture_hue_shift[0]:
            hue = _rand(*dr.fixture_hue_shift)
            try:
                prim = self._fixed_asset.prim if hasattr(self._fixed_asset, 'prim') else None
                if prim is not None:
                    _randomize_prim_materials(prim, hue_shift=hue, sat_mul=0.6, val_mul=0.8)
            except Exception:
                pass

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1.) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self._get_target_gear_base_offset()[0]

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
        # (a) get position vector to target
        bad_envs = env_ids.clone()
        ik_attempt = 0

        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.hand_down_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = fixed_tip_pos.clone()
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            # (b) get random orientation facing down
            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            self.hand_down_euler[bad_envs, ...] = hand_down_euler
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )

            # (c) iterative IK Method
            self.ctrl_target_fingertip_midpoint_pos[bad_envs, ...] = above_fixed_pos[bad_envs, ...]
            self.ctrl_target_fingertip_midpoint_quat[bad_envs, ...] = hand_down_quat[bad_envs, :]

            pos_error, aa_error = self.set_pos_inverse_kinematics(env_ids=bad_envs)
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            # Check IK succeeded for all envs, otherwise try again for those envs
            if bad_envs.shape[0] == 0:
                break

            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            )

            ik_attempt += 1

        self.step_sim_no_action()

        # Add flanking gears after servo (so arm doesn't move them).
        if self.cfg_task.name == "gear_mesh" and self.cfg_task.add_flanking_gears:
            small_gear_state = self._small_gear_asset.data.default_root_state.clone()[env_ids]
            small_gear_state[:, 0:7] = fixed_state[:, 0:7]
            small_gear_state[:, 7:] = 0.0  # vel
            self._small_gear_asset.write_root_pose_to_sim(small_gear_state[:, 0:7], env_ids=env_ids)
            self._small_gear_asset.write_root_velocity_to_sim(small_gear_state[:, 7:], env_ids=env_ids)
            self._small_gear_asset.reset()

            large_gear_state = self._large_gear_asset.data.default_root_state.clone()[env_ids]
            large_gear_state[:, 0:7] = fixed_state[:, 0:7]
            large_gear_state[:, 7:] = 0.0  # vel
            self._large_gear_asset.write_root_pose_to_sim(large_gear_state[:, 0:7], env_ids=env_ids)
            self._large_gear_asset.write_root_velocity_to_sim(large_gear_state[:, 7:], env_ids=env_ids)
            self._large_gear_asset.reset()

        # (3) Randomize asset-in-gripper location.
        # flip gripper z orientation
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat,
            t1=self.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros_like(self.fingertip_midpoint_pos),
        )

        # get default gripper in asset transform
        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        # Add asset in hand randomization
        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        if self.cfg_task.name == "gear_mesh":
            self.held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]

        held_asset_pos_noise = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        self.held_asset_pos_noise = self.held_asset_pos_noise @ torch.diag(held_asset_pos_noise)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=self.identity_quat,
            t2=self.held_asset_pos_noise,
        )

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        #  Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
        self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.ctrl_target_gripper_dof_pos = 0.0
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.raw_actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        # Reset encoder visual cache (Δv = v_t - prev_v_t needs clean start)
        if self._encoder is not None:
            self._encoder.reset_cache(env_ids)
        # Back out what actions should be for initial state.
        # Relative position to bolt tip.
        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        pos_actions = self.fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        # Relative yaw to bolt.
        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )

        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.fingertip_midpoint_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )

        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self._set_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
