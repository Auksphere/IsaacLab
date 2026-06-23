# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg

from .factory_tasks_cfg import ASSET_DIR, FactoryTask, GearMesh, NutThread, PegInsert, USBInsert

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "engagement_state": 1,
    "fingertip_dir_xy": 2,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}

FORCE_DIM_CFG = {
    "force_torque": 6,   # 6-DoF wrist FT (3 force + 3 torque), EMA-filtered & normalized
}

CONTACT_DIM_CFG = {
    "contact_force_state": 4,   # FREE / CONTACT_UP / CONTACT_DOWN / CONTACT_STEADY
    "engagement_state": 1,
    "keypoint_dist": 1,
}

# Force/torque filtering parameters (aligned with mujoco_menagerie config).
# Force normalisation: raw fingertip sum (gripper hold ~30N + contact ~0-20N)
# is divided by this value.  30N balances pre-differencing DC (~1.2) against
# post-differencing contact dynamics (~0.17).  (ema_alpha / baseline_fz_n /
# deadzone_n are unused since the filtering pipeline was simplified.)
FT_FILTER_CFG = {
    "ema_alpha": 0.3,
    "baseline_fz_n": -9.0,
    "deadzone_n": 1.0,
    "clip_n": 30.0,  # normalisation divisor
    "history_len": 5,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class DomainRandCfg:
    """Per-episode domain randomization for sim-to-real transfer.

    Applied during ``randomize_initial_state()``.  Each episode draws fresh
    parameters from uniform ranges.  Set a range to [0, 0] to disable that axis.
    """

    # ── Visual: lighting ──
    dome_light_intensity: list = [0.7, 1.3]       # multiplier on default intensity
    key_light_intensity: list = [0.8, 1.2]


    # ── Visual: camera ──
    camera_pos_noise: list = [0.00, 0.00, 0.00]  # metres, added to eye_local per episode
    camera_target_noise: list = [0.005, 0.005, 0.005]

    # ── Physics: mass ──
    held_mass_scale: list = [0.8, 1.2]             # peg mass multiplier
    fixed_mass_scale: list = [0.9, 1.1]            # hole fixture mass multiplier

    # ── Physics: friction ──
    held_friction: list = [0.6, 1.4]               # static/dynamic friction multiplier
    fixed_friction: list = [0.7, 1.3]

    # ── Physics: robot dynamics ──
    joint_damping_scale: list = [0.8, 1.2]         # multiplier on default kd
    joint_stiffness_scale: list = [0.9, 1.1]       # multiplier on default kp

    # ── Physics: peg/hole clearance ──
    peg_scale: list = [0.98, 1.00]                 # uniform scale on peg mesh
    hole_scale: list = [1.00, 1.02]                # uniform scale on hole fixture

    # ── Visual: material appearance ──
    peg_hue_shift: list = [0.0, 0.0]               # hue rotation in degrees (0-360)
    peg_saturation: list = [0.7, 1.3]              # saturation multiplier
    peg_value: list = [0.7, 1.3]                   # brightness multiplier
    table_roughness: list = [0.3, 0.9]             # roughness for table surface
    table_metallic: list = [0.0, 0.3]              # metallic for table surface
    table_color_shift: list = [0.0, 0.0]           # RGB shift on table base color
    floor_hue_shift: list = [0.0, 0.0]             # floor hue rotation
    fixture_hue_shift: list = [0.0, 0.0]           # hole fixture hue rotation

    # ── Full material replacement ──
    material_full_replace: bool = True          # If True: completely swap materials per-episode


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [0.005, 0.005, 0.005]
    rot_action_threshold = [0.097, 0.097, 0.097]

    reset_joints = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null = 10.0
    kd_null = 6.3246


@configclass
class FactoryEnvCfg(DirectRLEnvCfg):
    decimation = 8
    action_space = 6

    # ── Render quality (applied via carb settings before first render) ──
    # render_mode:
    #   "RaytracedLighting"  – default RTX, ~25ms/camera-frame (best quality)
    #   "RasterizedLighting" – pure rasterisation, ~3-8ms/camera-frame (fastest)
    #   "PathTracing"        – full path tracing (too slow for RL)
    render_mode: str = "RasterizedLighting"
    rtx_ao: bool = False            # ambient occlusion
    rtx_shadows: bool = False       # ray-traced shadows
    rtx_reflections: bool = False   # reflections
    rtx_gi: bool = False            # global illumination
    rtx_spp: int = 1               # samples per pixel (lower = faster)
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 21
    state_space = 72
    obs_order: list = ["fingertip_dir_xy", "engagement_state", "fingertip_quat", "ee_linvel", "ee_angvel"]  # SANITY: 19D, XY dir+eng
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]

    task_name: str = "peg_insert"  # peg_insert, gear_mesh, nut_thread
    task: FactoryTask = FactoryTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    domain_rand: DomainRandCfg = DomainRandCfg()
    ctrl: CtrlCfg = CtrlCfg()

    # Optional sensor configs. None means sensor disabled (backward-compatible).
    tiled_camera_left: TiledCameraCfg | None = None
    tiled_camera_right: TiledCameraCfg | None = None
    contact_sensor_fingertip: ContactSensorCfg | None = None
    contact_sensor_held_asset: ContactSensorCfg | None = None

    # Encoder checkpoint (Stage F). When set, policy obs = [bottleneck(256D), proprio(20D)].
    encoder_checkpoint: str = ""
    encoder_backbone: str = ""

    episode_length_s = 10.0  # Probably need to override.
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=87,
                velocity_limit=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=12,
                velocity_limit=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit=40.0,
                velocity_limit=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )


@configclass
class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 10.0


@configclass
class FactoryTaskGearMeshCfg(FactoryEnvCfg):
    task_name = "gear_mesh"
    task = GearMesh()
    episode_length_s = 20.0


@configclass
class FactoryTaskNutThreadCfg(FactoryEnvCfg):
    task_name = "nut_thread"
    task = NutThread()
    episode_length_s = 30.0


@configclass
class FactoryTaskPegInsertVisionCfg(FactoryTaskPegInsertCfg):
    """PegInsert with dual wrist cameras, wrist FT sensor, and richer privileged labels.

    Aligned with mujoco_menagerie config (assembly_gym_env.yaml):
    - Cameras: 224×224, 80° FOV, hand_camera1/hand_camera2 positioning from panda.xml
    - Force/torque: 6-DoF wrist FT via force_sensor body, EMA/baseline/deadzone/clip
    - FT history: 5 frames
    - Contact-phase labels: FREE/RIGID_CONTACT/COMPLIANT_CONTACT
    """

    # ── Cameras (aligned with mujoco hand_camera1 / hand_camera2) ──
    # MuJoCo: cameras are children of <body name="hand">, pos is body-local.
    #   hand_camera1: pos=(-0.10, 0, 0.07)  euler=(-π, -π/4, -π/2)  → quat (-0.2706,-0.6533,0.6533,-0.2706)
    #   hand_camera2: pos=( 0.10, 0, 0.07)  euler=( π,  π/4,  π/2)  → quat ( 0.2706, 0.6533,0.6533,-0.2706)
    #
    # ⚠️  TiledCamera.offset below is relative to the *env prim* (prim_path parent),
    #     NOT the hand body — it is only an initial pose. The actual hand-relative
    #     tracking is done by _update_camera_poses() in factory_env.py via
    #     set_world_poses(hand_pos + quat_apply(hand_quat, body_local_offset)).
    #     If you need to change the camera position, edit _update_camera_poses().
    # FOV = 69° (Intel RealSense D435 RGB, full-frame resize mode).
    #   FOV = 2·arctan(aperture / (2·focal))
    #   → focal = 20.955 / (2·tan(69°/2)) = 20.955 / (2·tan(34.5°)) ≈ 15.2 mm
    # Original mujoco setting was 80° (focal=12.5 mm).

    tiled_camera_left: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CameraLeft",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.08, -0.01, 0.08), rot=(-0.2706, -0.6533, 0.6533, -0.2706), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.2, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 2.0)
        ),
        width=224, height=224,
    )

    tiled_camera_right: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CameraRight",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.08, 0.01, 0.08), rot=(0.2706, 0.6533, 0.6533, -0.2706), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.2, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 2.0)
        ),
        width=224, height=224,
    )

    # ── Wrist force/torque sensor ──
    # Uses panda body "force_sensor" (body index 8) with get_link_incoming_joint_force().
    # Processing pipeline (aligned with mujoco gym_env.py:1117-1130):
    #   raw(6D) → EMA(α=0.3) → baseline(fz -= -9.0N) → deadzone(1.0N) → clip(50.0N) → normalize(/50.0)
    # ContactSensor is NOT used for FT — it's for future contact detection if needed.
    contact_sensor_fingertip: ContactSensorCfg | None = None
    contact_sensor_held_asset: ContactSensorCfg | None = None

    # Policy obs: encoder mode adds bottleneck; debug mode uses privileged state.
    obs_order: list = [
        "fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel",
    ]

    # Critic state: matches sanity_check checkpoint (43D = 37 + 6 actions).
    state_order: list = [
        "fingertip_pos", "fingertip_quat", "ee_linvel", "ee_angvel",
        "joint_pos", "held_pos", "held_pos_rel_fixed", "held_quat",
        "fixed_pos", "fixed_quat",
    ]


@configclass
class FactoryTaskPegInsertEncoderCfg(FactoryTaskPegInsertVisionCfg):
    """Vision env + frozen pretrained encoder for policy observations.

    Policy obs = [bottleneck(256D), proprio(20D), prev_actions(6D)] = 282D.
    Encoder is loaded during FactoryEnv.__init__.

    When ``encoder_debug_state_policy`` is True, the actor receives the same
    privileged state as the critic (bypassing the encoder).  This is a sanity-
    check mode to verify that the environment / RL pipeline has no bugs before
    debugging encoder-feature quality.
    """

    encoder_checkpoint: str = "/workspace/isaaclab/output/pretrain/pretrained_encoder_best.pt"
    encoder_backbone: str = "dinov2_vits14_attn"
    encoder_debug_state_policy: bool = False
    encoder_ablate_bottleneck: bool = False
    curriculum_pos_alpha_delay_steps: float = 655360
    curriculum_pos_alpha_decay_steps: float = 40960
    curriculum_pos_alpha_success_threshold: float = 0.90
    encoder_feed_task_pred: bool = True


@configclass
class FactoryTaskGearMeshEncoderCfg(FactoryTaskGearMeshCfg):
    """GearMesh with dual wrist cameras, wrist FT sensor, and frozen encoder.
    Same vision+FT setup as PegInsertEncoderCfg, different task (gear_mesh).
    """
    # ── Cameras ──
    tiled_camera_left: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CameraLeft",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.08, -0.01, 0.08), rot=(-0.2706, -0.6533, 0.6533, -0.2706), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.2, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 2.0)
        ),
        width=224, height=224,
    )
    tiled_camera_right: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CameraRight",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.02, 0.0, 0.02), rot=(0.7071, -0.7071, 0.0, 0.0), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.2, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 2.0)
        ),
        width=224, height=224,
    )

    # ── Wrist force/torque sensor ──
    contact_sensor_fingertip: ContactSensorCfg | None = None
    contact_sensor_held_asset: ContactSensorCfg | None = None

    # ── Obs/State orders (same as peg insert vision) ──
    obs_order: list = [
        "fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel",
    ]
    state_order: list = [
        "fingertip_pos", "fingertip_quat", "ee_linvel", "ee_angvel",
        "joint_pos", "held_pos", "held_pos_rel_fixed", "held_quat",
        "fixed_pos", "fixed_quat",
    ]

    # ── Encoder config ──
    encoder_checkpoint: str = "/workspace/isaaclab/output/pretrain/pretrained_encoder_best.pt"
    encoder_backbone: str = "dinov2_vits14_attn"
    encoder_debug_state_policy: bool = False
    encoder_ablate_bottleneck: bool = False
    curriculum_pos_alpha_delay_steps: float = 655360
    curriculum_pos_alpha_decay_steps: float = 163840
    curriculum_pos_alpha_success_threshold: float = 0.90
    encoder_feed_task_pred: bool = False


@configclass
class FactoryTaskPegInsertMonoEncoderCfg(FactoryTaskPegInsertEncoderCfg):
    """PegInsert with monocular camera + frozen encoder.

    Same as stereo encoder but right camera disabled.
    """

    tiled_camera_right: TiledCameraCfg | None = None
    encoder_mono: bool = False


@configclass
class FactoryTaskUSBInsertEncoderCfg(FactoryTaskPegInsertEncoderCfg):
    """USB-A insertion with encoder.  Same vision + FT setup, different mesh pair."""

    task_name: str = "usb_insert"
    task: FactoryTask = USBInsert()
