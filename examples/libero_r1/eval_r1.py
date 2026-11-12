"""R1-C closed-loop evaluator on LIBERO (spec sections 2.3, 2.4, 3.4, 4.3).

Arms (phase 1):
  vla        D: fine-tuned VLA, 1 sample per query, execute 3 steps, re-query.
  va_oracle  C: VA K=32 samples every 8 steps, cluster, <=6 candidates, pick the
                one closest to the VLA reference trajectory, execute 8 steps.
  va_random  A: VA K=32 samples every 8 steps, execute sample 0 (clustering is
                still computed, for logging only).
  va_jev     B: phase 2, selector interface reserved (traj_select.JevSelector).

Runs in the LIBERO env from the repo root:
  python -m examples.libero_r1.eval_r1 --arm va_oracle --suite libero_goal \
      --va_uri va --vla_uri vla --ns_port 9091 --out_root results/R1/R1C

Resumable: episodes already in <out>/episodes.csv are skipped; query lines of
unfinished episodes are dropped at start-up.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from shm_transport import setup_log_level

from . import libero_common as LC
from . import traj_select as TS
from .draw import VideoWriter, video_frame
from .policy_client import Policy, draw_ref_cam, noise_streams, ref_cam_streams

EP_FIELDS = ["suite", "task_id", "episode_id", "instruction", "arm", "success", "steps",
             "n_queries", "n_decisions", "wall_time_s", "seed"]
D_ARM_STEPS_PER_QUERY = 3   # existing evaluator: execute 3 steps then re-query
ARMS = ["vla", "va_oracle", "va_random", "va_jev"]


def episode_seed(suite: str, task_id: int, episode_id: int) -> int:
    return 100000 * LC.SUITE_INDEX[suite] + 1000 * task_id + episode_id


def r4(x):
    return np.round(np.asarray(x, np.float64), 4).tolist()


def _fsync(fp):
    fp.flush()
    os.fsync(fp.fileno())


class ResumableLog:
    """episodes.csv (one row per finished episode) + queries.jsonl.

    An episode's query lines are buffered in memory and written right before its
    CSV row, so a crash can at worst leave query lines without a CSV row; those
    are removed at start-up and the episode is re-run."""

    def __init__(self, out_dir: str, with_queries: bool):
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "episodes.csv")
        self.jsonl_path = os.path.join(out_dir, "queries.jsonl") if with_queries else None
        self.done = set()
        if os.path.exists(self.csv_path):
            with open(self.csv_path, newline="") as fp:
                for row in csv.DictReader(fp):
                    self.done.add((int(row["task_id"]), int(row["episode_id"])))
        else:
            with open(self.csv_path, "w", newline="") as fp:
                csv.DictWriter(fp, fieldnames=EP_FIELDS).writeheader()
                _fsync(fp)
        if self.jsonl_path and os.path.exists(self.jsonl_path):
            kept, dropped = [], 0
            with open(self.jsonl_path) as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    try:
                        q = json.loads(line)
                    except json.JSONDecodeError:
                        dropped += 1
                        continue
                    if (int(q["task_id"]), int(q["episode_id"])) in self.done:
                        kept.append(line if line.endswith("\n") else line + "\n")
                    else:
                        dropped += 1
            if dropped:
                tmp = self.jsonl_path + ".tmp"
                with open(tmp, "w") as fp:
                    fp.writelines(kept)
                    _fsync(fp)
                os.replace(tmp, self.jsonl_path)
                print(f"[RESUME] dropped {dropped} query lines of unfinished episodes")
        print(f"[RESUME] {len(self.done)} finished episodes found in {self.csv_path}")

    def commit(self, row: dict, query_lines):
        if self.jsonl_path:
            with open(self.jsonl_path, "a") as fp:
                for q in query_lines:
                    fp.write(json.dumps(q, ensure_ascii=False) + "\n")
                _fsync(fp)
        with open(self.csv_path, "a", newline="") as fp:
            csv.DictWriter(fp, fieldnames=EP_FIELDS).writerow(row)
            _fsync(fp)
        self.done.add((int(row["task_id"]), int(row["episode_id"])))


class Killer:
    """--debug_kill_at_query: hard-exit (no cleanup) mid-episode to test resume."""

    def __init__(self, n):
        self.n = n
        self.count = 0

    def tick(self):
        self.count += 1
        if self.n and self.count >= self.n:
            print(f"[DEBUG] killing the process at query #{self.count} (simulated crash)", flush=True)
            os._exit(17)


def _cams(env):
    Ka, Ta = LC.camera_KT(env.sim, "agentview")
    Ke, Te = LC.camera_KT(env.sim, "robot0_eye_in_hand")
    return (Ka, Ke), (Ta, Te)


def run_episode_vla(env, obs, instruction, vla: Policy, seed, max_steps, vid, killer):
    _va_rng, vla_rng = noise_streams(seed)
    _va_cam_rng, vla_cam_rng = ref_cam_streams(seed)
    t, q, done = 0, 0, False
    while not done and t < max_steps:
        frame = LC.obs_libero2apt(obs, time=t, env=env)
        out = vla.sample(frame, instruction, vla.noise(vla_rng, 1), draw_ref_cam(vla_cam_rng))
        killer.tick()
        poses, grip = out["poses"][0], out["grip"][0]
        q += 1
        for s in range(D_ARM_STEPS_PER_QUERY):
            obs, done = LC.step_with_pose(env, obs, poses[s], grip[s])
            t += 1
            if vid is not None:
                Ks, wcTs = _cams(env)
                vid.write(video_frame(obs, Ks, wcTs, poses[:, :3, 3], points_only=True,
                                      text=f"t={t} q={q}"))
            if done or t >= max_steps:
                break
    return dict(success=int(done), steps=t, n_queries=q, n_decisions=0), []


def run_episode_va(env, obs, instruction, selector: TS.Selector, va: Policy, vla, seed, max_steps,
                   vid, killer, ids):
    va_rng, vla_rng = noise_streams(seed)
    va_cam_rng, vla_cam_rng = ref_cam_streams(seed)
    t, q, n_dec, done = 0, 0, 0, False
    prev_sel_pos = None
    lines = []
    while not done and t < max_steps:
        frame = LC.obs_libero2apt(obs, time=t, env=env)
        out = va.sample(frame, instruction, va.noise(va_rng, TS.K_SAMPLES), draw_ref_cam(va_cam_rng))
        killer.tick()
        pos, grip, poses = out["pos"], out["grip"], out["poses"]
        cand = TS.make_candidates(pos, grip)
        ctx = TS.SelectionContext(cand=cand, pos=pos, grip=grip, instruction=instruction)
        vla_out = None
        if selector.needs_vla_ref:
            vla_out = vla.sample(frame, instruction, vla.noise(vla_rng, 1), draw_ref_cam(vla_cam_rng))
            ctx.ref_pos, ctx.ref_grip = vla_out["pos"][0], vla_out["grip"][0]
        sel = selector.select(ctx)
        if len(cand.candidates) >= 2:
            n_dec += 1
        k = sel.sample_idx
        sel_pos = pos[k]

        line = dict(ids)
        line.update(step=t, query_idx=q, n_clusters=cand.n_clusters, cluster_sizes=cand.cluster_sizes,
                    candidates=[dict(cluster_size=c.size, center_idx=c.center, pos=r4(pos[c.center]),
                                     gripper=r4(grip[c.center])) for c in cand.candidates],
                    selected_candidate=sel.candidate_idx, selected_sample_idx=k,
                    dist_selected_to_prev=TS.dist_selected_to_prev(prev_sel_pos, sel_pos),
                    sampling_time_s=out["rpc_time_s"], va_infer_time_s=out["infer_time_s"],
                    va_ref_cam=out["ref_cam"])
        if k not in [c.center for c in cand.candidates]:
            line.update(executed_pos=r4(pos[k]), executed_gripper=r4(grip[k]))
        if vla_out is not None:
            line.update(vla_time_s=vla_out["rpc_time_s"], vla_ref_cam=vla_out["ref_cam"], vla_ref_pos=r4(ctx.ref_pos),
                        vla_ref_gripper=r4(ctx.ref_grip))
        line.update(sel.info)
        lines.append(line)

        for s in range(TS.REPLAN_EVERY):
            obs, done = LC.step_with_pose(env, obs, poses[k, s], grip[k, s])
            t += 1
            if vid is not None:
                Ks, wcTs = _cams(env)
                vid.write(video_frame(obs, Ks, wcTs, pos[k, s:], text=f"t={t} q={q} nc={cand.n_clusters} sel={sel.candidate_idx}"))
            if done or t >= max_steps:
                break
        prev_sel_pos = sel_pos
        q += 1
    return dict(success=int(done), steps=t, n_queries=q, n_decisions=n_dec), lines


def main(args):
    setup_log_level("WARNING")
    if args.arm == "va_jev":
        raise SystemExit("va_jev is phase 2 and not implemented yet (spec section 2.5).")
    suite_name = args.suite
    assert suite_name in LC.SUITES, suite_name
    max_steps = LC.MAX_STEPS[suite_name]
    if args.max_steps_override:
        # spec section 8, ruling 8: D-arm extra check with a raised step cap only
        assert args.arm == "vla", "--max_steps_override is only defined for the D arm (spec 8.8)"
        assert args.run_name and args.run_name != args.arm, \
            "use a separate --run_name (e.g. vla_cap800) so the main D-arm results are not mixed"
        max_steps = int(args.max_steps_override)
        print(f"[INFO] max steps overridden: {LC.MAX_STEPS[suite_name]} -> {max_steps}")

    va = vla = None
    selector = None
    if args.arm == "vla":
        vla = Policy(args.vla_uri, args.ns_port, args.ns_host)
        assert vla.meta["train_stage"] == 1, vla.meta
    else:
        selector = TS.SELECTORS[args.arm]()
        va = Policy(args.va_uri, args.ns_port, args.ns_host)
        assert va.meta["train_stage"] == 0, f"VA server must hold a Stage-0 ckpt: {va.meta}"
        assert va.meta["ta"] == TS.TRAJ_LEN, va.meta
        if selector.needs_vla_ref:
            vla = Policy(args.vla_uri, args.ns_port, args.ns_host)
            assert vla.meta["train_stage"] == 1, vla.meta
    print(f"[INFO] VA meta: {va.meta if va else None}")
    print(f"[INFO] VLA meta: {vla.meta if vla else None}")

    out_dir = os.path.join(args.out_root, args.run_name or args.arm, suite_name)
    log = ResumableLog(out_dir, with_queries=(args.arm != "vla"))
    with open(os.path.join(out_dir, f"run_meta_{time.strftime('%Y%m%d-%H%M%S')}.json"), "w") as fp:
        json.dump(dict(args=vars(args), va=va.meta if va else None, vla=vla.meta if vla else None,
                       max_steps=max_steps), fp, indent=2)
    killer = Killer(args.debug_kill_at_query)

    suite = LC.get_suite(suite_name)
    task_end = min(args.task_end, suite.n_tasks)
    task_ids = args.tasks if args.tasks else list(range(args.task_start, task_end))
    assert all(0 <= t < suite.n_tasks for t in task_ids), task_ids
    n_succ = n_run = 0
    for task_id in task_ids:
        episodes = list(range(args.episode_start, args.episode_start + args.episodes_per_task))
        todo = [e for e in episodes if (task_id, e) not in log.done]
        for e in episodes:
            if e not in todo:
                print(f"[SKIP] {suite_name} task {task_id} episode {e} already done")
        if not todo:
            continue
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, instruction = LC.make_env(task)
        print(f"[INFO] {suite_name} task {task_id}: {instruction}")
        try:
            for e in todo:
                seed = episode_seed(suite_name, task_id, e)
                t0 = time.time()
                obs = LC.reset_to_state(env, init_states[e])
                vid = None
                if not args.no_video and e < args.n_videos:
                    vid = VideoWriter(os.path.join(out_dir, "videos", f"task{task_id:02d}", f"ep{e:02d}.mp4"))
                ids = dict(suite=suite_name, task_id=task_id, episode_id=e)
                try:
                    if args.arm == "vla":
                        res, lines = run_episode_vla(env, obs, instruction, vla, seed, max_steps, vid, killer)
                    else:
                        res, lines = run_episode_va(env, obs, instruction, selector, va, vla, seed,
                                                    max_steps, vid, killer, ids)
                finally:
                    if vid is not None:
                        vid.close()
                row = dict(ids, instruction=instruction, arm=args.arm, wall_time_s=round(time.time() - t0, 3),
                           seed=seed, **res)
                log.commit(row, lines)
                n_succ += res["success"]
                n_run += 1
                print(f"[EP] {suite_name} t{task_id} e{e} success={res['success']} steps={res['steps']} "
                      f"queries={res['n_queries']} | this run {n_succ}/{n_run}", flush=True)
        finally:
            env.close()
    print(f"[DONE] {out_dir}")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--suite", required=True, choices=LC.SUITES)
    p.add_argument("--task_start", type=int, default=0)
    p.add_argument("--task_end", type=int, default=10, help="exclusive")
    p.add_argument("--tasks", nargs="*", type=int, default=None,
                   help="explicit task ids (overrides --task_start/--task_end), e.g. 0 3 7")
    p.add_argument("--max_steps_override", type=int, default=0,
                   help="D arm only, spec 8.8 extra check (e.g. 800); requires a separate --run_name")
    p.add_argument("--episode_start", type=int, default=0)
    p.add_argument("--episodes_per_task", type=int, default=50)
    p.add_argument("--out_root", default="results/R1/R1C")
    p.add_argument("--run_name", default="", help="sub-directory under out_root (default: the arm name)")
    p.add_argument("--va_uri", default="va")
    p.add_argument("--vla_uri", default="vla")
    p.add_argument("--ns_host", default="localhost")
    p.add_argument("--ns_port", type=int, default=9091)
    p.add_argument("--n_videos", type=int, default=3, help="record episodes with id < n_videos")
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--debug_kill_at_query", type=int, default=0,
                   help="smoke test 10 only: hard-exit at the N-th policy query of this process")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
