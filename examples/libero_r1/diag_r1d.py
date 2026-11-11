"""R1-D: open-loop multimodality diagnostic of the VA (spec sections 2.2, 3.5, 4.2).

For each (suite, task, state, VA checkpoint): reset to the fixed initial state,
run 10 dummy steps, draw 64 VA samples, cluster them at 0.04 / 0.08 / 0.12 m,
and compare with one VLA reference trajectory (true instruction) on the same
observation. The VA server's weights are hot-swapped per checkpoint
(`load_weights`), so one VA server and one VLA server are enough.

Runs in the LIBERO env from the repo root:
  python -m examples.libero_r1.diag_r1d --va_uri va --vla_uri vla --ns_port 9091 \
      --va_ckpts /abs/ckpt_0010000.pt /abs/ckpt_0020000.pt ... --out_dir results/R1/R1D

Resumable: rows already in r1d_rows.csv are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

from shm_transport import setup_log_level

from . import libero_common as LC
from . import traj_select as TS
from .draw import draw_r1d_figure
from .policy_client import Policy, draw_ref_cam, noise_streams, ref_cam_streams

N_SAMPLES = 64
FIELDS = ["suite", "task_id", "state_id", "ckpt_iter", "instruction",
          "n_clusters_004", "n_clusters_008", "n_clusters_012", "cluster_sizes_008",
          "covered", "min_dist_to_vla", "vla_cluster_rank", "vla_cluster_share",
          "seed", "ckpt_path", "va_ref_cam", "vla_ref_cam"]


def diag_seed(suite: str, task_id: int, state_id: int) -> int:
    return 10000 * LC.SUITE_INDEX[suite] + 100 * task_id + state_id


def main(args):
    setup_log_level("WARNING")
    va = Policy(args.va_uri, args.ns_port, args.ns_host)
    vla = Policy(args.vla_uri, args.ns_port, args.ns_host)
    assert vla.meta["train_stage"] == 1, vla.meta
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "r1d_rows.csv")
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fp:
            for r in csv.DictReader(fp):
                done.add((r["suite"], int(r["task_id"]), int(r["state_id"]), int(r["ckpt_iter"])))
    else:
        with open(csv_path, "w", newline="") as fp:
            csv.DictWriter(fp, fieldnames=FIELDS).writeheader()
    ref_dir = os.path.join(args.out_dir, "vla_ref")
    os.makedirs(ref_dir, exist_ok=True)
    vis_iters = set(int(x) for x in args.vis_iters)
    with open(os.path.join(args.out_dir, f"run_meta_{time.strftime('%Y%m%d-%H%M%S')}.json"), "w") as fp:
        json.dump(dict(args=vars(args), vla=vla.meta), fp, indent=2)

    for ckpt in args.va_ckpts:
        meta = va.load_weights(ckpt)
        assert meta["train_stage"] == 0, meta
        it = int(meta["current_iters"])
        print(f"[INFO] VA checkpoint {ckpt} (iter {it})")
        for suite_name in args.suites:
            suite = LC.get_suite(suite_name)
            for task_id in args.tasks:
                todo = [s for s in args.states if (suite_name, task_id, s, it) not in done]
                if not todo:
                    continue
                task = suite.get_task(task_id)
                init_states = suite.get_task_init_states(task_id)
                env, instruction = LC.make_env(task)
                try:
                    for state_id in todo:
                        seed = diag_seed(suite_name, task_id, state_id)
                        va_rng, vla_rng = noise_streams(seed)
                        va_cam_rng, vla_cam_rng = ref_cam_streams(seed)
                        va_cam, vla_cam = draw_ref_cam(va_cam_rng), draw_ref_cam(vla_cam_rng)
                        obs = LC.reset_to_state(env, init_states[state_id])
                        frame = LC.obs_libero2apt(obs, time=0, env=env)

                        ref_file = os.path.join(ref_dir, f"{suite_name}_t{task_id:02d}_s{state_id}.npz")
                        if os.path.exists(ref_file):
                            z = np.load(ref_file)
                            ref_pos, ref_grip = z["pos"], z["grip"]
                            assert str(z["ref_cam"]) == vla_cam, f"stale VLA reference cache {ref_file}"
                        else:
                            r = vla.sample(frame, instruction, vla.noise(vla_rng, 1), vla_cam)
                            ref_pos, ref_grip = r["pos"][0], r["grip"][0]
                            np.savez(ref_file, pos=ref_pos, grip=ref_grip, seed=seed, ref_cam=vla_cam,
                                     vla_ckpt=str(vla.meta.get("ckpt_path")))

                        out = va.sample(frame, instruction, va.noise(va_rng, N_SAMPLES), va_cam)
                        stats, cand = TS.r1d_analyse(out["pos"], out["grip"], ref_pos, ref_grip)
                        row = dict(suite=suite_name, task_id=task_id, state_id=state_id, ckpt_iter=it,
                                   instruction=instruction, seed=seed, ckpt_path=ckpt,
                                   va_ref_cam=va_cam, vla_ref_cam=vla_cam, **stats)
                        with open(csv_path, "a", newline="") as fp:
                            csv.DictWriter(fp, fieldnames=FIELDS).writerow(row)
                        done.add((suite_name, task_id, state_id, it))
                        print(f"[R1D] it={it} {suite_name} t{task_id} s{state_id} "
                              f"nc(.04/.08/.12)={stats['n_clusters_004']}/{stats['n_clusters_008']}/"
                              f"{stats['n_clusters_012']} covered={stats['covered']} "
                              f"min_d={stats['min_dist_to_vla']:.3f}", flush=True)

                        if args.vis_all or it in vis_iters:
                            rank_of = np.empty(N_SAMPLES, dtype=int)
                            for r_i, c in enumerate(cand.clusters):
                                rank_of[c.members] = r_i
                            K, wcT = LC.camera_KT(env.sim, "agentview")
                            img = draw_r1d_figure(frame["agentview"]["data"]["color"], K, wcT, out["pos"],
                                                  rank_of, ref_pos,
                                                  title=f"{suite_name} t{task_id} s{state_id} it{it} nc={cand.n_clusters}")
                            vdir = os.path.join(args.out_dir, "vis", f"iter{it:07d}")
                            os.makedirs(vdir, exist_ok=True)
                            cv2.imwrite(os.path.join(vdir, f"{suite_name}_t{task_id:02d}_s{state_id}.png"), img)
                finally:
                    env.close()
    print(f"[DONE] {csv_path}")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--va_uri", default="va")
    p.add_argument("--vla_uri", default="vla")
    p.add_argument("--ns_host", default="localhost")
    p.add_argument("--ns_port", type=int, default=9091)
    p.add_argument("--va_ckpts", nargs="+", required=True,
                   help="checkpoint paths as seen by the VA server (its JSON must sit next to it)")
    p.add_argument("--suites", nargs="+", default=LC.SUITES, choices=LC.SUITES)
    p.add_argument("--tasks", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--states", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--out_dir", default="results/R1/R1D")
    p.add_argument("--vis_iters", nargs="*", default=["40000"], help="checkpoint iterations to draw figures for")
    p.add_argument("--vis_all", action="store_true", help="draw figures for every checkpoint (smoke test)")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
