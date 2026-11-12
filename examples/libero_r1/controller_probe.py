"""R1-Y: read the LIBERO OSC controller configuration and check the commanded-target
formula against the controller's own goal (spec 12.2). LIBERO env, repo root:

  python -m examples.libero_r1.controller_probe --raw_root <LIBERO>/libero/datasets \
      --out results/R1/R1Y/controller_config.json

Replays the first demo of libero_goal task 0 (actions mode, no-ops skipped, as the
converter does) and after every step compares
  formula target (clip + scale + left-multiplied rotation + zero-rotation hold), and
  controller target (controller.goal_pos / goal_ori after env.step, converted from the
  grip-site frame to our body-orientation convention).
Writes the controller configuration (output limits etc.) and the comparison statistics.
"""
from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

from . import libero_common as LC
from .convert_libero import is_noop, natural_key


def rot_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main(args):
    suite = LC.get_suite(args.suite)
    task = suite.get_task(args.task)
    env, _ = LC.make_env(task)
    info = LC.controller_info(env)
    raw = os.path.join(args.raw_root, args.suite, f"{task.name}_demo.hdf5")
    with h5py.File(raw, "r") as f:
        dk = sorted(f["data"].keys(), key=natural_key)[0]
        states, actions = f["data"][dk]["states"][()], f["data"][dk]["actions"][()]
    env.reset()
    obs = env.set_init_state(states[0])
    ctrl = env.env.robots[0].controller
    dpos, dang, offs, n_zero_rot, prev, prev_goal = [], [], [], 0, None, None
    C0 = None
    for t in range(len(actions)):
        if is_noop(actions[t], prev):
            continue
        prev = actions[t]
        ee = LC.ee_pose_from_obs(obs)
        C = LC.site_to_body_offset(env, obs)
        C0 = C if C0 is None else C0
        offs.append(rot_angle(C0, C))
        goal_f, _ = LC.commanded_target_formula(ee, actions[t], info, prev_goal)
        n_zero_rot += int(np.allclose(actions[t][3:6], 0))
        obs, _r, _d, _i = env.step(actions[t].tolist())
        goal_c_pos = np.asarray(ctrl.goal_pos, float)
        goal_c_rot = np.asarray(ctrl.goal_ori, float).reshape(3, 3) @ C
        dpos.append(float(np.linalg.norm(goal_f[:3, 3] - goal_c_pos)))
        dang.append(rot_angle(goal_f[:3, :3], goal_c_rot))
        prev_goal = goal_c_rot
        if len(dpos) >= args.max_steps:
            break
    env.close()
    out = dict(suite=args.suite, task=args.task, task_name=task.name, demo=dk, controller=info,
               n_steps_checked=len(dpos), n_zero_rotation_actions=n_zero_rot,
               formula_vs_controller_pos_diff_m=dict(max=max(dpos), median=float(np.median(dpos))),
               formula_vs_controller_rot_diff_deg=dict(max=max(dang), median=float(np.median(dang))),
               site_to_body_offset_drift_deg_max=max(offs))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_root", required=True)
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--out", default="results/R1/R1Y/controller_config.json")
    main(p.parse_args())
