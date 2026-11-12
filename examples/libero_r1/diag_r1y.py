"""R1-Y diagnostics and the R1b checkpoint selection (spec section 12). APT env, repo
root, one GPU, APT_DATA_LOC(_HOST) set; no inference server.

  # Y1: VLA vs commanded targets rebuilt from the raw demo actions (X2's 200 samples)
  python -m examples.libero_r1.diag_r1y y1 --vla checkpoints/hf/apt_vla_ftlibero/ckpt_latest.pt \
      --raw_root <LIBERO>/libero/datasets --controller_json results/R1/R1Y/controller_config.json
  # Y2: frame-rotation augmentation on / off for the 40k VA
  python -m examples.libero_r1.diag_r1y y2 --va checkpoints/APT/va_libero4_r1/ckpt_0040000.pt
  # Y3: position error of the clean trajectory implied by the noise prediction, per timestep bin
  python -m examples.libero_r1.diag_r1y y3 --models va10k=... va20k=... va30k=... va40k=... vla=... \
      --raw_root ... --controller_json ...
  # 12.3.2: data check of the NEW (desired-pose) data with the VLA
  python -m examples.libero_r1.diag_r1y datacheck --out_dir results/R1/R1Y/datacheck
  # 12.3.4: post-training check and checkpoint choice on the NEW (desired-pose) data
  python -m examples.libero_r1.diag_r1y posttrain --models va40k=... va60k=... --out_dir results/R1/R1Y/posttrain

Samples: y1/y2/y3 reuse X2's cached samples (--samples, default results/R1/R1X/x2_samples.pt),
i.e. the same 200 LiberoGoal samples (seed 0, augmentation off). posttrain draws 200 new
samples (seed 0) from the data the current APT_DATA_LOC points at, cached in its out_dir.

Commanded targets for the OLD data (Y1, Y3-VLA) are rebuilt with
libero_common.commanded_target_formula: clip(action) x controller action_scale (read from
--controller_json, written by controller_probe on the LIBERO side), position added to the
pre-step EE position, rotation left-multiplied, orientation held when the rotation command
is exactly zero (robosuite OSC behaviour). Frame t's target belongs to the action executed
at frame t; the future window follows dataset_base `future_desired_time` (t, t+1, ..., t+31,
clamped at the episode end). Gripper target = (1 - clip(a7)) / 2.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import h5py
import numpy as np
import torch

from examples.libero_r1 import diag_r1x_x2 as X2
from examples.libero_r1.traj_select import motion_stats

BINS = X2.BINS
T_PER_BIN = X2.T_PER_BIN
STEP_IDS = X2.STEP_IDS
K_VA = X2.K_VA


# ── commanded targets for old-format episodes ───────────────────────────────

def _natural_key(s):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _is_noop(action, prev_action=None, threshold=1e-4):
    if prev_action is None:
        return bool(np.linalg.norm(action[:-1]) < threshold)
    return bool(np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1])


def commanded_episode(episode_file, raw_root, info, cache):
    """(T,4,4) commanded EE targets and (T,) gripper targets for every frame of a converted
    episode, rebuilt from the raw demo actions."""
    if episode_file in cache:
        return cache[episode_file]
    from examples.libero_r1.libero_common import commanded_target_formula
    with h5py.File(episode_file, "r") as h5:
        ee = h5["ee_pose"][()].astype(np.float64)                   # (T,4,4)
        attrs = dict(h5.attrs)
    assert attrs.get("replay") == "actions", f"{episode_file}: needs actions-mode data"
    raw = os.path.join(raw_root, attrs["suite"], f"{attrs['task_name']}_demo.hdf5")
    with h5py.File(raw, "r") as f:
        actions = f["data"][attrs["demo_key"]]["actions"][()]
    kept, prev = [], None
    for a in actions:
        if attrs.get("filter_noops", True) and _is_noop(a, prev):
            continue
        prev = a
        kept.append(a)
    assert len(kept) == len(ee), f"{episode_file}: {len(kept)} kept actions vs {len(ee)} frames"
    poses, grips, prev_rot = [], [], None
    for t in range(len(ee)):
        g, gr = commanded_target_formula(ee[t], kept[t], info, prev_rot)
        poses.append(g)
        grips.append(gr)
        prev_rot = g[:3, :3]
    out = (np.stack(poses), np.asarray(grips))
    cache[episode_file] = out
    return out


def desired_future(rec, raw_root, info, cache, Ta=32):
    """(Ta, 1, 17) commanded-target future for a cached sample, as a float32 tensor."""
    poses, grips = commanded_episode(rec["_episode_file"], raw_root, info, cache)
    t = int(round(float(np.asarray(rec["timestamps"]).reshape(-1)[-1])))
    idx = np.clip(t + np.arange(Ta), 0, len(poses) - 1)       # future_desired_time = t + [0..Ta-1]
    fut = np.concatenate([poses[idx].reshape(Ta, 16), grips[idx][:, None]], axis=-1)[:, None]
    return torch.from_numpy(fut.astype(np.float32))


def load_info(path):
    with open(path) as fp:
        j = json.load(fp)
    return j.get("controller", j)


# ── model calls ──────────────────────────────────────────────────────────────

def train_branch(model, d, gt_future, i, b, rot=None, return_x0=False):
    """Training-branch loss (and optionally x0) on T_PER_BIN copies of one sample with
    timesteps from bin b; seeds as in X2."""
    from apt.infer.planner import _repeat_batch
    lo, hi = BINS[b]
    g = torch.Generator().manual_seed(1000 * i + b)
    ts = torch.randint(lo, hi, (T_PER_BIN,), generator=g)
    torch.manual_seed(1000 * i + b)
    dd = dict(d)
    dd["gt_future_ee_states"] = gt_future.unsqueeze(0).to(d["obs_rgbs"].device)
    with torch.inference_mode():
        vl_in, vl_out = model.vlm(obs_rgbs=dd["obs_rgbs"], obs_masks=dd["obs_masks"],
                                  obs_norm_xys=dd["obs_norm_xys"], obs_extrinsics=dd["obs_extrinsics"],
                                  prompt_text=dd["prompt_text"], fp16=True)
        r = _repeat_batch(dd, T_PER_BIN)
        out = model.actor(vl_inputs=_repeat_batch(vl_in, T_PER_BIN), vl_outputs=_repeat_batch(vl_out, T_PER_BIN),
                          current_ee_pose=r["current_ee_pose"], action_ref_pose=r["action_ref_pose"],
                          history_ee_states=r["history_ee_states"], gt_future_ee_states=r["gt_future_ee_states"],
                          valid_ee_mask=r["valid_ee_mask"], inference=False, fp16=True,
                          train_timesteps=ts.to(dd["obs_rgbs"].device), random_frame_rotation=rot,
                          return_x0=return_x0)
    return out


def sample_with_rotation(model, d, Kp, seed, rot):
    """Like X2.predict (20 steps, bf16) but with the mean-pose frame rotation forced on/off."""
    from apt.infer.planner import _repeat_batch
    with torch.inference_mode():
        vl_in, vl_out = model.vlm(obs_rgbs=d["obs_rgbs"], obs_masks=d["obs_masks"],
                                  obs_norm_xys=d["obs_norm_xys"], obs_extrinsics=d["obs_extrinsics"],
                                  prompt_text=d["prompt_text"], fp16=True)
        Ta, nee = d["gt_future_ee_states"].shape[1:3]
        torch.manual_seed(seed)  # the random rotation draws from torch's generator
        noise = torch.from_numpy(np.random.default_rng(seed).standard_normal(
            (Kp, nee, Ta, model.actor.act_dim)).astype(np.float32)).to(d["obs_rgbs"].device)
        rep = _repeat_batch(d, Kp)
        pred = model.actor(vl_inputs=_repeat_batch(vl_in, Kp), vl_outputs=_repeat_batch(vl_out, Kp),
                           current_ee_pose=rep["current_ee_pose"], action_ref_pose=rep["action_ref_pose"],
                           history_ee_states=rep["history_ee_states"], gt_future_ee_states=rep["gt_future_ee_states"],
                           valid_ee_mask=rep["valid_ee_mask"], inference=True, fp16=True, initial_noise=noise,
                           random_frame_rotation=rot)
    pred = pred.float()[:, :, 0]
    return (pred[..., :16].reshape(Kp, Ta, 4, 4)[..., :3, 3].double().cpu().numpy(),
            pred[..., 16].double().cpu().numpy())


def step_and_grip(ppos, pgrip, gt_fut):
    """Step errors / gripper MAE of the best (min mean position error) trajectory vs gt_fut (Ta,1,17)."""
    gpos = gt_fut[:, 0, :16].reshape(-1, 4, 4)[:, :3, 3].double().numpy()
    ggrip = gt_fut[:, 0, 16].double().numpy()
    se = np.linalg.norm(ppos - gpos[None], axis=-1)
    e = se.mean(axis=1)
    b = int(np.argmin(e))
    out = {f"err_step{k}": float(se[b, k - 1]) for k in STEP_IDS}
    out.update(err_best=float(e[b]), grip_mae=float(np.abs(pgrip[b] - ggrip).mean()),
               grip_gt_mean=float(ggrip.mean()), grip_pred_mean=float(pgrip[b].mean()))
    return out


def med(rows, k):
    return float(np.median([r[k] for r in rows]))


def load(ckpt, device):
    from apt.infer.planner import load_model
    model, _ = load_model(ckpt, device)
    return model


# ── Y1 ───────────────────────────────────────────────────────────────────────

def run_y1(args, samples, device):
    info = load_info(args.controller_json)
    cache = {}
    model = load(args.vla, device)
    assert model.r1_meta["train_stage"] == 1
    rows, loss_rows = [], []
    for i, rec in enumerate(samples):
        d = X2.batch1(rec, device)
        fut_des = desired_future(rec, args.raw_root, info, cache)
        ppos, pgrip = X2.predict(model, d, 1, seed=i)
        r_des = step_and_grip(ppos, pgrip, fut_des)
        r_act = step_and_grip(ppos, pgrip, rec["gt_future_ee_states"])
        row = dict(sample=i, **{f"desired_{k}": v for k, v in r_des.items()},
                   **{f"actual_{k}": v for k, v in r_act.items()},
                   commanded_lead_step1=float(np.linalg.norm(
                       fut_des[0, 0, :16].reshape(4, 4)[:3, 3].double().numpy() - X2.cur_pos(rec))))
        rows.append(row)
        for b in range(len(BINS)):
            _l, met = train_branch(model, d, fut_des, i, b)
            loss_rows.append(dict(sample=i, bin=f"{BINS[b][0]}-{BINS[b][1] - 1}", **{k: float(v) for k, v in met.items()}))
        if (i + 1) % 50 == 0:
            print(f"[Y1] {i + 1}/{len(samples)}", flush=True)
    write_rows(os.path.join(args.out_dir, "y1_rows.csv"), rows)
    write_rows(os.path.join(args.out_dir, "y1_loss.csv"), loss_rows)
    summ = dict(controller=info,
                desired={**{f"err_step{k}_median": med(rows, f"desired_err_step{k}") for k in STEP_IDS},
                         "grip_mae_median": med(rows, "desired_grip_mae"),
                         "grip_gt_mean": float(np.mean([r["desired_grip_gt_mean"] for r in rows])),
                         "grip_pred_mean": float(np.mean([r["desired_grip_pred_mean"] for r in rows]))},
                actual={f"err_step{k}_median": med(rows, f"actual_err_step{k}") for k in STEP_IDS},
                commanded_lead_step1_median=med(rows, "commanded_lead_step1"),
                loss_by_bin=bin_means(loss_rows))
    a_holds = summ["desired"]["err_step1_median"] < 0.01
    summ["A_holds"] = bool(a_holds)
    dump(os.path.join(args.out_dir, "y1_summary.json"), summ)
    print("[Y1] step errors vs commanded targets (cm): " +
          ", ".join(f"{k}:{summ['desired'][f'err_step{k}_median']*100:.2f}" for k in STEP_IDS))
    print("[Y1] (vs actual poses, cm): " + ", ".join(f"{k}:{summ['actual'][f'err_step{k}_median']*100:.2f}" for k in STEP_IDS))
    print(f"[Y1] gripper MAE vs commanded median {summ['desired']['grip_mae_median']:.3f}; "
          f"commanded lead at step 1 median {summ['commanded_lead_step1_median']*100:.2f} cm")
    print(f"[Y1-VERDICT] step-1 error median vs commanded = {summ['desired']['err_step1_median']*100:.2f} cm "
          f"(< 1 cm) -> {'A HOLDS' if a_holds else 'A NOT CONFIRMED'}")


# ── Y2 ───────────────────────────────────────────────────────────────────────

def run_y2(args, samples, device):
    model = load(args.va, device)
    assert model.r1_meta["train_stage"] == 0
    res, rows = {}, []
    for rot in (True, False):
        tag = "random_rotation" if rot else "no_rotation"
        loss_rows, samp_rows = [], []
        for i, rec in enumerate(samples):
            d = X2.batch1(rec, device)
            for b in range(len(BINS)):
                _l, met = train_branch(model, d, rec["gt_future_ee_states"], i, b, rot=rot)
                loss_rows.append(dict(sample=i, bin=f"{BINS[b][0]}-{BINS[b][1] - 1}", **{k: float(v) for k, v in met.items()}))
            ppos, pgrip = sample_with_rotation(model, d, K_VA, seed=i, rot=rot)
            m = motion_stats(ppos, pgrip, cur_pos=X2.cur_pos(rec))
            sg = step_and_grip(ppos, pgrip, rec["gt_future_ee_states"])
            samp_rows.append(dict(sample=i, adj_disp_median=m["adj_disp_median"], err_best=sg["err_best"]))
            rows.append(dict(condition=tag, sample=i, adj_disp_median=m["adj_disp_median"], err_best=sg["err_best"]))
        res[tag] = dict(loss_by_bin=bin_means(loss_rows), adj_disp_median=med(samp_rows, "adj_disp_median"),
                        err_best_median=med(samp_rows, "err_best"))
        write_rows(os.path.join(args.out_dir, f"y2_loss_{tag}.csv"), loss_rows)
        print(f"[Y2] {tag}: {json.dumps(res[tag])}", flush=True)
    write_rows(os.path.join(args.out_dir, "y2_samples.csv"), rows)
    lo_r = res["random_rotation"]["loss_by_bin"]["0-19"]["pos_loss"]
    lo_n = res["no_rotation"]["loss_by_bin"]["0-19"]["pos_loss"]
    c1 = lo_n > 1.3 * lo_r
    c2 = res["random_rotation"]["adj_disp_median"] <= 0.015
    res["verdict"] = dict(low_bin_pos_loss_no_over_random=lo_n / lo_r, cond_loss=bool(c1), cond_sampling=bool(c2),
                          rotation_mismatch=bool(c1 or c2))
    dump(os.path.join(args.out_dir, "y2_summary.json"), res)
    print(f"[Y2-VERDICT] low-bin pos_loss no/random = {lo_n / lo_r:.2f} (> 1.30?) ; adjacent-step median with "
          f"random rotation = {res['random_rotation']['adj_disp_median']*100:.2f} cm (<= 1.5 cm?) -> "
          f"{'ROTATION MISMATCH (stop, report)' if (c1 or c2) else 'not triggered'}")


# ── Y3 ───────────────────────────────────────────────────────────────────────

def y3_table(model, samples, device, gt_fn):
    acc = {b: [] for b in range(len(BINS))}
    for i, rec in enumerate(samples):
        d = X2.batch1(rec, device)
        gt_fut = gt_fn(rec)
        gpos = gt_fut[:, 0, :16].reshape(-1, 4, 4)[:, :3, 3].double().numpy()
        for b in range(len(BINS)):
            _l, _m, x0 = train_branch(model, d, gt_fut, i, b, return_x0=True)
            xpos = x0.float()[:, :, 0, :16].reshape(T_PER_BIN, -1, 4, 4)[..., :3, 3].double().cpu().numpy()
            acc[b].append(float(np.linalg.norm(xpos - gpos[None], axis=-1).mean()))
    return {f"{lo}-{hi - 1}": float(np.mean(acc[b])) for b, (lo, hi) in enumerate(BINS)}


def run_y3(args, samples, device):
    info = load_info(args.controller_json) if args.controller_json else None
    cache = {}
    res = {}
    for item in args.models:
        label, ckpt = item.split("=", 1)
        model = load(ckpt, device)
        if model.r1_meta["train_stage"] == 1:
            assert info is not None and args.raw_root, "VLA in Y3 needs --raw_root and --controller_json"
            gt_fn = lambda rec: desired_future(rec, args.raw_root, info, cache)  # noqa: E731
            gt_kind = "commanded"
        else:
            gt_fn = lambda rec: rec["gt_future_ee_states"]  # noqa: E731
            gt_kind = "loader"
        res[label] = dict(gt=gt_kind, ckpt=ckpt, iter=model.r1_meta["current_iters"],
                          x0_pos_err_m_by_bin=y3_table(model, samples, device, gt_fn))
        print(f"[Y3] {label} ({gt_kind} GT): " +
              ", ".join(f"{k}: {v*100:.2f} cm" for k, v in res[label]["x0_pos_err_m_by_bin"].items()), flush=True)
        del model
        torch.cuda.empty_cache()
    dump(os.path.join(args.out_dir, "y3_summary.json"), res)


# ── 12.3.2 data check on the new data ────────────────────────────────────────

def run_datacheck(args, samples, device):
    model = load(args.vla, device)
    assert model.r1_meta["train_stage"] == 1
    rows = []
    for i, rec in enumerate(samples):
        d = X2.batch1(rec, device)
        ppos, pgrip = X2.predict(model, d, 1, seed=i)
        rows.append(dict(sample=i, **step_and_grip(ppos, pgrip, rec["gt_future_ee_states"])))
    write_rows(os.path.join(args.out_dir, "datacheck_rows.csv"), rows)
    res = {**{f"err_step{k}_median": med(rows, f"err_step{k}") for k in STEP_IDS},
           "grip_mae_median": med(rows, "grip_mae"),
           "grip_gt_mean": float(np.mean([r["grip_gt_mean"] for r in rows])),
           "grip_pred_mean": float(np.mean([r["grip_pred_mean"] for r in rows]))}
    res["pass"] = bool(res["err_step1_median"] < 0.01 and res["grip_mae_median"] < 0.1)
    dump(os.path.join(args.out_dir, "datacheck_summary.json"), res)
    print("[DATACHECK] VLA vs new-data ground truth, step errors (cm): " +
          ", ".join(f"{k}:{res[f'err_step{k}_median']*100:.2f}" for k in STEP_IDS))
    print(f"[DATACHECK] step-1 median {res['err_step1_median']*100:.2f} cm (< 1), gripper MAE median "
          f"{res['grip_mae_median']:.3f} (< 0.1) -> {'PASS' if res['pass'] else 'FAIL (stop, report)'}")


# ── 12.3.4 post-training check ───────────────────────────────────────────────

def run_posttrain(args, samples, device):
    res = {}
    for item in args.models:
        label, ckpt = item.split("=", 1)
        model = load(ckpt, device)
        assert model.r1_meta["train_stage"] == 0
        rows = []
        for i, rec in enumerate(samples):
            d = X2.batch1(rec, device)
            ppos, pgrip = X2.predict(model, d, K_VA, seed=i)
            m = motion_stats(ppos, pgrip, cur_pos=X2.cur_pos(rec))
            sg = step_and_grip(ppos, pgrip, rec["gt_future_ee_states"])
            rows.append(dict(sample=i, adj_disp_median=m["adj_disp_median"], **sg))
        write_rows(os.path.join(args.out_dir, f"posttrain_{label}.csv"), rows)
        r = dict(ckpt=ckpt, iter=model.r1_meta["current_iters"], adj_disp_median=med(rows, "adj_disp_median"),
                 err_best_median=med(rows, "err_best"),
                 **{f"err_step{k}_median": med(rows, f"err_step{k}") for k in STEP_IDS},
                 grip_mae_median=med(rows, "grip_mae"),
                 x0_pos_err_m_by_bin=y3_table(model, samples, device, lambda rec: rec["gt_future_ee_states"]))
        r["pass"] = bool(r["adj_disp_median"] <= 0.015 and r["err_best_median"] <= 0.04)
        res[label] = r
        print(f"[POSTTRAIN] {label} (iter {r['iter']}): adjacent-step median {r['adj_disp_median']*100:.2f} cm (<= 1.5), "
              f"best-of-8 error median {r['err_best_median']*100:.2f} cm (<= 4) -> {'PASS' if r['pass'] else 'FAIL'}",
              flush=True)
        del model
        torch.cuda.empty_cache()
    order = args.prefer  # spec 12.3.4: 60k first, then 40k
    chosen = next((lab for lab in order if lab in res and res[lab]["pass"]), None)
    res["selected"] = None if chosen is None else dict(label=chosen, ckpt=res[chosen]["ckpt"], iter=res[chosen]["iter"])
    dump(os.path.join(args.out_dir, "posttrain_summary.json"), res)
    if chosen is None:
        print("[POSTTRAIN-SELECT] no checkpoint passes -> STOP and report (attach Y3 bins above)")
    else:
        print(f"[POSTTRAIN-SELECT] use {chosen}: {res[chosen]['ckpt']}")
        with open(os.path.join(args.out_dir, "selected_ckpt.txt"), "w") as fp:
            fp.write(os.path.abspath(res[chosen]["ckpt"]) + "\n")


# ── utils / main ─────────────────────────────────────────────────────────────

def bin_means(rows):
    out = {}
    for lo, hi in BINS:
        key = f"{lo}-{hi - 1}"
        sel = [r for r in rows if r["bin"] == key]
        out[key] = {k: float(np.mean([r[k] for r in sel])) for k in ("pos_loss", "rot_loss", "openness_loss", "total_loss")}
    return out


def write_rows(path, rows):
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def dump(path, obj):
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=2)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda:0"
    if args.cmd in ("posttrain", "datacheck"):
        samples = X2.build_samples(os.path.join(args.out_dir, "x2_samples.pt"))
        with h5py.File(samples[0]["_episode_file"], "r") as h5:
            has_des = "ee_pose_desired" in h5
        assert has_des, f"{args.cmd} must run on the desired-pose (R1b) data; check APT_DATA_LOC"
    else:
        samples = torch.load(args.samples, weights_only=False)
    print(f"[R1-Y] {args.cmd}: {len(samples)} samples")
    {"y1": run_y1, "y2": run_y2, "y3": run_y3, "datacheck": run_datacheck,
     "posttrain": run_posttrain}[args.cmd](args, samples, device)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["y1", "y2", "y3", "datacheck", "posttrain"])
    p.add_argument("--vla", default="checkpoints/hf/apt_vla_ftlibero/ckpt_latest.pt")
    p.add_argument("--va", default="checkpoints/APT/va_libero4_r1/ckpt_0040000.pt")
    p.add_argument("--models", nargs="*", default=[], help="label=ckpt (y3, posttrain)")
    p.add_argument("--prefer", nargs="*", default=["va60k", "va40k"], help="posttrain selection order")
    p.add_argument("--raw_root", default="")
    p.add_argument("--controller_json", default="results/R1/R1Y/controller_config.json")
    p.add_argument("--samples", default="results/R1/R1X/x2_samples.pt")
    p.add_argument("--out_dir", default="results/R1/R1Y")
    main(p.parse_args())
