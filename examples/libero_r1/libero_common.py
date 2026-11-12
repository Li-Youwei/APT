"""LIBERO helpers shared by the R1 evaluator, diagnostic and data converter.

Runs in the LIBERO conda env (no torch / apt imports). The observation and
action conversions are verbatim copies of `examples/libero/eval_libero_pred.py`
(`obs_libero2apt`, `action_from_apt_to_libero`); env construction and the dummy
action reproduce OpenVLA-OFT's `get_libero_env` / `get_libero_dummy_action`
(the dependency of the original evaluator), so this package does not need
openvla-oft on PYTHONPATH.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np

IMAGE_RESOLUTION = 256
NUM_WAIT_STEPS = 10            # dummy steps after reset, not counted (spec 2.3)

# Spec 2.2 / 2.3 suite order (suite index used in the seeds) and max steps.
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_INDEX = {s: i for i, s in enumerate(SUITES)}
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def get_suite(suite_name: str):
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[suite_name]()


def make_env(task, resolution: int = IMAGE_RESOLUTION):
    """Same as OpenVLA-OFT get_libero_env: bddl env at `resolution`, env.seed(0)."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)  # OpenVLA-OFT: seed affects object positions even with fixed init states
    return env, task.language


def reset_to_state(env, init_state, n_wait: int = NUM_WAIT_STEPS):
    """reset -> set init state -> n_wait dummy steps (not counted). Returns obs."""
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(n_wait):
        obs, _r, _d, _i = env.step(DUMMY_ACTION)
    return obs


def camera_KT(sim, cam: str, res: int = IMAGE_RESOLUTION) -> Tuple[np.ndarray, np.ndarray]:
    """(K for res x res, wcT = ^world_cam T) as used by the APT obs frames."""
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
    return (get_camera_intrinsic_matrix(sim, cam, res, res), get_camera_extrinsic_matrix(sim, cam))


def ee_pose_from_obs(obs) -> np.ndarray:
    import robosuite.utils.transform_utils as T
    ee = np.eye(4)
    ee[:3, 3] = obs["robot0_eef_pos"]
    ee[:3, :3] = T.quat2mat(obs["robot0_eef_quat"])
    return ee


def obs_libero2apt(obs: dict, time: int, env) -> dict:
    """Verbatim copy of examples/libero/eval_libero_pred.py::obs_libero2apt."""
    import robosuite.utils.transform_utils as T
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

    ee_pose = np.eye(4)
    ee_pose[:3, 3] = obs["robot0_eef_pos"]
    ee_pose[:3, :3] = T.quat2mat(obs["robot0_eef_quat"])

    gripper_state = obs["robot0_gripper_qpos"]      # (2,)
    open_gripper_qpos, close_gripper_qpos = 0.04, 0.0
    gripper = (gripper_state[0] - close_gripper_qpos) / (open_gripper_qpos - close_gripper_qpos)

    # Cameras: MuJoCo renders upside-down — flip H to bring image right-side up.
    e2h_rgb = np.ascontiguousarray(obs["agentview_image"][::-1])
    eih_rgb = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    e2h_pose = get_camera_extrinsic_matrix(env.sim, "agentview")
    eih_pose = get_camera_extrinsic_matrix(env.sim, "robot0_eye_in_hand")
    e2h_K = get_camera_intrinsic_matrix(env.sim, "agentview", IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    eih_K = get_camera_intrinsic_matrix(env.sim, "robot0_eye_in_hand", IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    return {
        "ee_pose":   ee_pose,
        "gripper":   gripper,
        "timestamp": time,
        "agentview": {
            "model":  "pinhole",
            "camera": {"height": IMAGE_RESOLUTION, "width": IMAGE_RESOLUTION,
                       "K": e2h_K.flatten().tolist()},
            "data":   {"color": e2h_rgb, "wcT": e2h_pose,
                       "seg": None, "depth": None},
        },
        "eye_in_hand": {
            "model":  "pinhole",
            "camera": {"height": IMAGE_RESOLUTION, "width": IMAGE_RESOLUTION,
                       "K": eih_K.flatten().tolist()},
            "data":   {"color": eih_rgb, "wcT": eih_pose,
                       "seg": None, "depth": None},
        },
    }


def action_from_apt_to_libero(
    future_ee_poses: np.ndarray,     # (T, 4, 4) or (4, 4)
    future_grippers: np.ndarray,     # (T,) or scalar
    current_eef_pos: np.ndarray,     # (3,)
    current_eef_rotmat: np.ndarray,  # (3, 3)
    robosuite_action_scale: np.ndarray,
):
    """Verbatim copy of examples/libero/eval_libero_pred.py::action_from_apt_to_libero."""
    from scipy.spatial.transform import Rotation as R
    delta_pos = future_ee_poses[..., :3, 3] - current_eef_pos
    delta_rotmat = future_ee_poses[..., :3, :3] @ np.linalg.inv(current_eef_rotmat)
    delta_ori = R.from_matrix(delta_rotmat).as_rotvec()
    delta_pose_scaled = np.concatenate([delta_pos, delta_ori], axis=-1) / robosuite_action_scale
    binary = (future_grippers > 0.8).astype(np.float32)
    delta_gripper = 1 - 2 * binary
    return np.concatenate([delta_pose_scaled, delta_gripper[..., None]], axis=-1)


def step_with_pose(env, obs, target_pose: np.ndarray, target_grip: float):
    """Execute one env step toward an absolute EE target. Returns (obs, done)."""
    import robosuite.utils.transform_utils as T
    action = action_from_apt_to_libero(
        future_ee_poses=target_pose,
        future_grippers=np.asarray(target_grip),
        current_eef_pos=obs["robot0_eef_pos"],
        current_eef_rotmat=T.quat2mat(obs["robot0_eef_quat"]),
        robosuite_action_scale=env.env.robots[0].controller.action_scale,
    )
    obs, _r, done, _info = env.step(action.tolist())
    return obs, bool(done)


# ── commanded (desired) EE targets of robosuite's OSC_POSE controller (spec 12.2 / 12.3) ──
#
# robosuite v1.4.1 OSC.set_goal (verified against the source, see codex_instructions):
#   a = clip(action[:6], input_min, input_max); delta = a * action_scale (+ output transform, 0 here)
#   goal_pos = ee_pos + delta[:3]                (ee_pos = grip-site position, same as obs robot0_eef_pos)
#   goal_ori = R(delta[3:6]) @ ee_ori_mat        (ee_ori_mat = grip-SITE orientation), but goal_ori is
#              only updated when some rotation component is non-zero; otherwise the previous goal is kept
#   (position / orientation limits clip the goal when configured; None in LIBERO by default)
# Our EE pose uses obs robot0_eef_quat, the hand BODY orientation, which differs from the site by a
# constant rotation C = R_site^T R_body; a goal in our convention is goal_ori_site @ C.


def controller_info(env) -> dict:
    ctrl = env.env.robots[0].controller

    def arr(x):
        return None if x is None else np.asarray(x, dtype=float).tolist()

    scale = ctrl.action_scale
    if scale is None:  # robosuite computes it lazily on the first scale_action call
        scale = np.abs(np.asarray(ctrl.output_max) - np.asarray(ctrl.output_min)) / \
            np.abs(np.asarray(ctrl.input_max) - np.asarray(ctrl.input_min))
    return dict(
        controller=type(ctrl).__name__, eef_site=getattr(ctrl, "eef_name", None),
        input_min=arr(ctrl.input_min), input_max=arr(ctrl.input_max),
        output_min=arr(ctrl.output_min), output_max=arr(ctrl.output_max), action_scale=arr(scale),
        use_delta=bool(getattr(ctrl, "use_delta", True)), use_ori=bool(getattr(ctrl, "use_ori", True)),
        position_limits=arr(getattr(ctrl, "position_limits", None)),
        orientation_limits=arr(getattr(ctrl, "orientation_limits", None)),
        interpolator=None if getattr(ctrl, "interpolator_pos", None) is None else type(ctrl.interpolator_pos).__name__,
        control_freq=float(getattr(env.env, "control_freq", float("nan"))),
    )


def site_to_body_offset(env, obs) -> np.ndarray:
    """C = R_site^T @ R_body for the controller's EE site and the obs EE body orientation."""
    import robosuite.utils.transform_utils as T
    ctrl = env.env.robots[0].controller
    sid = env.sim.model.site_name2id(ctrl.eef_name)
    R_site = np.asarray(env.sim.data.site_xmat[sid]).reshape(3, 3)
    return R_site.T @ T.quat2mat(obs["robot0_eef_quat"])


def commanded_target_formula(ee_pose: np.ndarray, action: np.ndarray, info: dict, prev_goal_rot=None):
    """Spec 12.2 formula with the controller's clipping and 'hold orientation on zero rotation'
    rule. ee_pose (4,4) in our convention (site position, body orientation). Returns (T_goal (4,4),
    gripper target in [0,1] with 1 = open)."""
    from scipy.spatial.transform import Rotation as R
    a = np.clip(np.asarray(action[:6], float), info["input_min"], info["input_max"])
    delta = a * np.asarray(info["action_scale"], float)
    goal = np.eye(4)
    goal[:3, 3] = ee_pose[:3, 3] + delta[:3]
    if np.any(~np.isclose(delta[3:6], 0.0)) or prev_goal_rot is None:
        goal[:3, :3] = R.from_rotvec(delta[3:6]).as_matrix() @ ee_pose[:3, :3]
    else:
        goal[:3, :3] = prev_goal_rot
    grip = float((1.0 - np.clip(action[6], -1.0, 1.0)) / 2.0)   # -1 (open) -> 1, +1 (close) -> 0
    return goal, grip
