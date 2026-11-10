"""Convert the official LIBERO demo HDF5 files into APT's per-episode HDF5 format
(spec section 3.2). Runs in the LIBERO env from the repo root.

Input : <raw_root>/<suite>/<task.name>_demo.hdf5   (official LIBERO datasets,
        `python benchmark_scripts/download_libero_datasets.py`)
Output: <out_root>/<suite>/<task.name>/demo_<k>.h5, one file per demo, with
          agentview/{rgb (JPEG, vlen uint8), K (3,3), pose (T,4,4)}
          eye_in_hand/{rgb, K, pose}
          ee_pose (T,4,4) world frame, gripper (T,) openness in [0,1],
          timestamp (T,) = 0,1,2,...   attrs: prompt_text, compress=True, ...
        <out_root>/<suite>_stats.json with demo counts.

Every frame is produced by the same `obs_libero2apt` packaging the evaluator
feeds to the policy (256x256, H-flipped images, camera K / extrinsics from the
sim, gripper = finger qpos / 0.04), so training and inference see identical
conventions. JPEG bytes are produced exactly like data_utils.h5io.jpeg_encode.

Default (spec section 7, ruling 1 -- OpenVLA regeneration, same data recipe as
`apt_vla_ftlibero`): `--replay actions` sets the first recorded state, steps
the recorded actions (frames = observations before each executed action),
skips no-op actions (OpenVLA rule; `--keep_noops` disables) and drops demos
that do not end in success. `--replay states` (re-render each recorded state)
is kept only for debugging; it is not the R1 recipe.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import cv2
import h5py
import numpy as np

from . import libero_common as LC


def jpeg_encode(image: np.ndarray) -> np.ndarray:
    # identical to data_utils.h5io.jpeg_encode (cv2 sees the RGB array as-is;
    # data_utils.h5io.jpeg_decode returns the same channel order)
    return np.frombuffer(cv2.imencode(".jpg", image)[1].data, dtype=np.uint8)


def is_noop(action, prev_action=None, threshold=1e-4) -> bool:
    """OpenVLA regenerate_libero_dataset.py rule."""
    if prev_action is None:
        return bool(np.linalg.norm(action[:-1]) < threshold)
    return bool(np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1])


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def write_episode(path: str, frames, prompt: str, attrs: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as h5:
        for cam in ("agentview", "eye_in_hand"):
            g = h5.create_group(cam)
            rgb = g.create_dataset("rgb", shape=(len(frames),), dtype=h5py.vlen_dtype(np.uint8))
            for i, f in enumerate(frames):
                rgb[i] = jpeg_encode(f[cam]["data"]["color"])
            g.create_dataset("K", data=np.asarray(frames[-1][cam]["camera"]["K"], np.float32).reshape(3, 3))
            g.create_dataset("pose", data=np.stack([f[cam]["data"]["wcT"] for f in frames]).astype(np.float32))
        h5.create_dataset("ee_pose", data=np.stack([f["ee_pose"] for f in frames]).astype(np.float32))
        h5.create_dataset("gripper", data=np.asarray([f["gripper"] for f in frames], np.float32))
        h5.create_dataset("timestamp", data=np.arange(len(frames), dtype=np.float32))
        h5.attrs["prompt_text"] = prompt
        h5.attrs["compress"] = True
        for k, v in attrs.items():
            h5.attrs[k] = v
    os.replace(tmp, path)


def replay_states(env, states, actions, filter_noops):
    frames, prev = [], None
    for t in range(len(states)):
        obs = env.set_init_state(states[t])
        if filter_noops:
            if is_noop(actions[t], prev):
                continue
            prev = actions[t]
        frames.append(LC.obs_libero2apt(obs, time=len(frames), env=env))
    # success check: last recorded state + last recorded action
    env.set_init_state(states[-1])
    _obs, _r, done, _i = env.step(actions[-1].tolist())
    return frames, bool(done)


def replay_actions(env, states, actions, filter_noops):
    env.reset()
    obs = env.set_init_state(states[0])
    frames, prev, done = [], None, False
    for t in range(len(actions)):
        if filter_noops:
            if is_noop(actions[t], prev):
                continue
            prev = actions[t]
        frames.append(LC.obs_libero2apt(obs, time=len(frames), env=env))
        obs, _r, done, _i = env.step(actions[t].tolist())
    return frames, bool(done)


def main(args):
    args.filter_noops = not args.keep_noops
    for suite_name in args.suites:
        suite = LC.get_suite(suite_name)
        stats = dict(suite=suite_name, replay=args.replay, filter_noops=args.filter_noops,
                     drop_failed=args.drop_failed, tasks={})
        stats_path = os.path.join(args.out_root, f"{suite_name}_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as fp:
                stats["tasks"] = json.load(fp).get("tasks", {})
        tasks = args.tasks if args.tasks else list(range(suite.n_tasks))
        for task_id in tasks:
            task = suite.get_task(task_id)
            raw = os.path.join(args.raw_root, suite_name, f"{task.name}_demo.hdf5")
            assert os.path.exists(raw), f"missing raw demo file {raw}"
            env, instruction = LC.make_env(task)
            out_dir = os.path.join(args.out_root, suite_name, task.name)
            st = dict(task_name=task.name, instruction=instruction, n_raw=0, n_written=0,
                      n_replay_failed=0, n_dropped=0, failed_demos=[], n_frames=0, n_frames_raw=0,
                     n_noop_frames_dropped=0)
            t0 = time.time()
            with h5py.File(raw, "r") as f:
                demo_keys = sorted(f["data"].keys(), key=natural_key)
                for k_i, dk in enumerate(demo_keys):
                    st["n_raw"] += 1
                    out_path = os.path.join(out_dir, f"demo_{k_i:03d}.h5")
                    states = f["data"][dk]["states"][()]
                    actions = f["data"][dk]["actions"][()]
                    st["n_frames_raw"] += len(states)
                    if args.replay == "states":
                        frames, ok = replay_states(env, states, actions, args.filter_noops)
                    else:
                        frames, ok = replay_actions(env, states, actions, args.filter_noops)
                    if not ok:
                        st["n_replay_failed"] += 1
                        st["failed_demos"].append(dk)
                    drop = (not ok) and (args.drop_failed or args.replay == "actions")
                    if drop:
                        st["n_dropped"] += 1
                        if os.path.exists(out_path):
                            os.remove(out_path)
                        continue
                    write_episode(out_path, frames, instruction,
                                  dict(suite=suite_name, task_id=task_id, task_name=task.name, demo_key=dk,
                                       replay=args.replay, filter_noops=bool(args.filter_noops),
                                       replay_success=bool(ok)))
                    st["n_written"] += 1
                    st["n_frames"] += len(frames)
                    st["n_noop_frames_dropped"] += len(actions) - len(frames) if args.filter_noops else 0
            env.close()
            st["seconds"] = round(time.time() - t0, 1)
            stats["tasks"][str(task_id)] = st
            print(f"[CONVERT] {suite_name} t{task_id} {task.name}: written {st['n_written']}/{st['n_raw']} "
                  f"replay_failed={st['n_replay_failed']} dropped={st['n_dropped']} ({st['seconds']}s)", flush=True)
            os.makedirs(args.out_root, exist_ok=True)
            with open(stats_path, "w") as fp:
                json.dump(stats, fp, indent=2)
        tot = {k: sum(t[k] for t in stats["tasks"].values())
               for k in ("n_raw", "n_written", "n_replay_failed", "n_dropped", "n_frames", "n_noop_frames_dropped")}
        stats["total"] = tot
        with open(stats_path, "w") as fp:
            json.dump(stats, fp, indent=2)
        print(f"[CONVERT] {suite_name} total: kept {tot['n_written']} / {tot['n_raw']} episodes, "
              f"dropped {tot['n_dropped']} (replay failed); {tot}")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_root", required=True, help="dir holding libero_spatial/, libero_object/, ... *_demo.hdf5")
    p.add_argument("--out_root", required=True)
    p.add_argument("--suites", nargs="+", default=LC.SUITES, choices=LC.SUITES)
    p.add_argument("--tasks", nargs="*", type=int, default=None, help="subset of task ids (default all)")
    p.add_argument("--replay", choices=["states", "actions"], default="actions")
    p.add_argument("--keep_noops", action="store_true", help="debug only: do not drop no-op actions")
    p.add_argument("--drop_failed", action="store_true",
                   help="states mode: also drop demos whose success check fails (actions mode always drops)")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
