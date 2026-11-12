"""R1-X / X1: inference-path control (spec section 9.2). LIBERO env, repo root,
with a VLA server (`--vla_uri`) and a Stage-0 VA server (`--va_uri`) running.
The VA server's weights are hot-swapped through the listed checkpoints.

  python -m examples.libero_r1.diag_r1x \
      --va_ckpts apt_va=$PWD/checkpoints/hf/apt_va/ckpt_latest.pt \
                 va10k=$PWD/checkpoints/APT/va_libero4_r1/ckpt_0010000.pt \
                 va40k=$PWD/checkpoints/APT/va_libero4_r1/ckpt_0040000.pt \
      --out_dir results/R1/R1X

States: libero_goal tasks 0-9, initial state 0, after 10 dummy steps.
Models: apt_vla_ftlibero (VLA server) + every --va_ckpts entry.
Paths : shared  = TrajPlanner.sample_trajs (one VLM pass, repeated K times,
                  numpy noise from default_rng(seed));
        replicated = the repository's original get_action(sample_num=K)
                  (observation replicated K times, VLA.forward, torch noise
                  seeded with `seed`).
Cameras: agentview, eye_in_hand (pinned).  K = 32.  seed = state index (task id).
Each (model, path, camera, state) writes one row of x1_rows.csv (resumable).
Figures: the model labelled `--vis_model` (default va40k), shared path,
agentview -> x1_vis/<model>_goal_t<task>.png, VLA (shared, agentview) sample 0
drawn as the white reference.
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
from .policy_client import REF_CAMS, Policy

K = 32
SUITE = "libero_goal"
STATE_ID = 0
PATHS = ("shared", "replicated")
FIELDS = ["model", "train_stage", "ckpt_iter", "path", "ref_cam", "suite", "task_id", "state_id", "seed",
          "instruction", "adj_disp_median", "adj_disp_p95", "adj_disp_max", "net_disp_median",
          "net_disp_first_last_median", "n_clusters_008", "grip_std_median", "infer_time_s", "ckpt_path"]


def collect_frames(tasks):
    suite = LC.get_suite(SUITE)
    out = {}
    for t in tasks:
        task = suite.get_task(t)
        env, instr = LC.make_env(task)
        try:
            obs = LC.reset_to_state(env, suite.get_task_init_states(t)[STATE_ID])
            frame = LC.obs_libero2apt(obs, time=0, env=env)
            Ka, wcTa = LC.camera_KT(env.sim, "agentview")
        finally:
            env.close()
        out[t] = dict(frame=frame, instr=instr, cur_pos=np.asarray(frame["ee_pose"])[:3, 3], K=Ka, wcT=wcTa)
    return out


def run_one(pol: Policy, path, frame, instr, seed, cam):
    if path == "shared":
        noise = pol.noise(np.random.default_rng(seed), K)
        return pol.sample(frame, instr, noise, cam)
    return pol.sample_legacy(frame, instr, K, seed, cam)


def main(args):
    setup_log_level("WARNING")
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "x1_rows.csv")
    if args.summarize_only:
        summarize(csv_path, args.out_dir)
        return
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fp:
            for r in csv.DictReader(fp):
                done.add((r["model"], r["path"], r["ref_cam"], int(r["task_id"])))
    else:
        with open(csv_path, "w", newline="") as fp:
            csv.DictWriter(fp, fieldnames=FIELDS).writeheader()

    vla = Policy(args.vla_uri, args.ns_port, args.ns_host)
    assert vla.meta["train_stage"] == 1, vla.meta
    va = Policy(args.va_uri, args.ns_port, args.ns_host) if args.va_ckpts else None
    models = [] if args.skip_vla else [("vla", None)]
    for item in args.va_ckpts:
        label, path = item.split("=", 1)
        models.append((label, path))
    frames = collect_frames(args.tasks)
    with open(os.path.join(args.out_dir, f"x1_run_meta_{time.strftime('%Y%m%d-%H%M%S')}.json"), "w") as fp:
        json.dump(dict(args=vars(args), vla=vla.meta), fp, indent=2)

    vla_ref = {}  # task -> (32, 3) VLA shared/agentview sample 0, for figures
    vis_dir = os.path.join(args.out_dir, "x1_vis")
    for label, ckpt in models:
        if ckpt is None:
            pol = vla
        else:
            pol = va
            meta = pol.load_weights(ckpt)
            assert meta["train_stage"] == 0, meta
        meta = pol.meta
        print(f"[X1] model {label}: {meta}")
        for path in args.paths:
            for cam in args.ref_cams:
                for t in args.tasks:
                    f = frames[t]
                    key = (label, path, cam, t)
                    need_vis = (label == args.vis_model and path == "shared" and cam == "agentview")
                    if key in done and not (label == "vla" and path == "shared" and cam == "agentview"
                                            and args.vis_model in [m for m, _ in models]):
                        continue
                    seed = t  # spec 9.2: seed = state index (states are tasks 0-9)
                    out = run_one(pol, path, f["frame"], f["instr"], seed, cam)
                    if label == "vla" and path == "shared" and cam == "agentview":
                        vla_ref[t] = out["pos"][0]
                    if key in done:
                        continue
                    st = TS.motion_stats(out["pos"], out["grip"], cur_pos=f["cur_pos"])
                    ncl = TS.make_candidates(out["pos"], out["grip"]).n_clusters
                    row = dict(model=label, train_stage=meta["train_stage"], ckpt_iter=meta.get("current_iters"),
                               path=path, ref_cam=cam, suite=SUITE, task_id=t, state_id=STATE_ID, seed=seed,
                               instruction=f["instr"], n_clusters_008=ncl, infer_time_s=round(out["infer_time_s"], 4),
                               ckpt_path=meta.get("ckpt_path"),
                               **{k: round(v, 6) for k, v in st.items()})
                    with open(csv_path, "a", newline="") as fp:
                        csv.DictWriter(fp, fieldnames=FIELDS).writerow(row)
                    done.add(key)
                    print(f"[X1] {label:<8} {path:<10} {cam:<11} t{t}: adj med/p95 = "
                          f"{st['adj_disp_median']*100:.2f}/{st['adj_disp_p95']*100:.2f} cm, net med = "
                          f"{st['net_disp_median']*100:.2f} cm, clusters = {ncl}, grip std = {st['grip_std_median']:.3f}",
                          flush=True)
                    if need_vis:
                        D = TS.pairwise_distances(out["pos"], out["grip"])
                        cl = TS.sort_clusters(D, TS.complete_linkage(D, TS.CLUSTER_THRESH))
                        rank = np.empty(K, dtype=int)
                        for r_i, c in enumerate(cl):
                            rank[c.members] = r_i
                        img = draw_r1d_figure(f["frame"]["agentview"]["data"]["color"], f["K"], f["wcT"], out["pos"],
                                              rank, vla_ref.get(t), title=f"{label} shared agentview goal t{t} nc={len(cl)}")
                        os.makedirs(vis_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(vis_dir, f"{label}_goal_t{t:02d}.png"), img)
    summarize(csv_path, args.out_dir)


def summarize(csv_path, out_dir):
    with open(csv_path, newline="") as fp:
        rows = list(csv.DictReader(fp))
    keys = ["adj_disp_median", "adj_disp_p95", "net_disp_median", "n_clusters_008", "grip_std_median"]
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["path"], r["ref_cam"]), []).append(r)
    out = []
    for (m, p, c), g in groups.items():
        rec = dict(model=m, path=p, ref_cam=c, n_states=len(g))
        for k in keys:
            rec["mean_" + k] = round(float(np.mean([float(x[k]) for x in g])), 6)
        out.append(rec)
    with open(os.path.join(out_dir, "x1_summary.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("[X1] summary (means over states; metres):")
    for r in out:
        print("   ", r)
    vla_rows = [r for r in rows if r["model"] == "vla"]
    if vla_rows:
        # spec section 10, ruling 1: speed-independent criterion, every VLA row must satisfy
        # p95 / median of adjacent-step displacement <= 2.0 and clusters at 0.08 m <= 3
        ratios = [float(r["adj_disp_p95"]) / max(float(r["adj_disp_median"]), 1e-12) for r in vla_rows]
        ncls = [int(r["n_clusters_008"]) for r in vla_rows]
        ok = all(x <= 2.0 for x in ratios) and all(n <= 3 for n in ncls)
        print(f"[X1-SMOKE-2] VLA rows: worst p95/median = {max(ratios):.2f} (<= 2.0), "
              f"max clusters = {max(ncls)} (<= 3) over {len(vla_rows)} rows -> "
              f"{'PASS' if ok else 'FAIL (diagnostic script is wrong)'}")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--va_uri", default="va")
    p.add_argument("--vla_uri", default="vla")
    p.add_argument("--ns_host", default="localhost")
    p.add_argument("--ns_port", type=int, default=9091)
    p.add_argument("--va_ckpts", nargs="*", default=[], help="label=path (path as seen by the VA server)")
    p.add_argument("--skip_vla", action="store_true", help="smoke only: do not run the VLA model")
    p.add_argument("--tasks", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--paths", nargs="+", default=list(PATHS), choices=PATHS)
    p.add_argument("--ref_cams", nargs="+", default=list(REF_CAMS), choices=REF_CAMS)
    p.add_argument("--vis_model", default="va40k")
    p.add_argument("--out_dir", default="results/R1/R1X")
    p.add_argument("--summarize_only", action="store_true",
                   help="recompute x1_summary.csv and the smoke check from existing x1_rows.csv (no servers)")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
