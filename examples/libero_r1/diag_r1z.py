"""R1-Z diagnostics (spec section 13). APT env, repo root, one GPU, APT_DATA_LOC pointing at
the NEW (commanded-target, R1b) data, APT_DATA_LOC_HOST set. No inference server.

  # Z1 + Z3 + Z4: per-suite checks of the VA (8 samples) and the VLA (1 sample), 200 samples each
  python -m examples.libero_r1.diag_r1z suites --va checkpoints/APT/va_libero4_r1b/ckpt_0060000.pt \
      --vla checkpoints/hf/apt_vla_ftlibero/ckpt_latest.pt --out_dir results/R1/R1Z
  # Z2: pos_loss and x0 position error at fixed diffusion timesteps (LiberoGoal samples of Z1)
  python -m examples.libero_r1.diag_r1z timesteps --va ... --vla ... --out_dir results/R1/R1Z
  # 13.3 post-training check of the overfit runs (100 samples of Goal task 1)
  python -m examples.libero_r1.diag_r1z overfit --models p2k=... p4k=... q2k=... --out_dir results/R1/R1Z/overfit

Definitions (all positions in metres in the CSV/JSON, the console prints cm):
  ground truth = the loader's future (commanded targets on the new data), 32 steps;
  adjacent-step displacement = |p[t+1] - p[t]| over the 32 predicted (or true) points;
    `*_pooled` = median / p95 over all steps of all samples (and all 8 VA trajectories),
    `*_of_samples` = median over samples of each sample's median (as in R1-Y posttrain);
  best-trajectory error = 32-step mean position error of the best of K (VA K = 8, VLA K = 1);
  step k error = position error at future step k of that best trajectory;
  clusters = complete-linkage clusters at 0.08 m of the K trajectories (traj_select);
  Z4 smoothing = per-axis centred moving average of the predicted POSITIONS over 7 steps
    (window truncated at both ends), orientation and gripper untouched, applied to the VA
    samples only; every metric is recomputed on the smoothed trajectories.
  Z2: for each sample and each fixed timestep t in {0,1,2,5,10,20,40,60,80,99}, 4 random
    noises (training branch, training-time frame randomisation, seeded) -> pos_loss and the
    position error of the clean trajectory reconstructed from the noise prediction.
  Z5 (spec 15.2, `equivariance`): 20 LiberoGoal samples (seed 0); at t in {5, 50, 95} the
    denoiser's noise prediction for a noisy trajectory vs for the same trajectory with its 32
    future tokens permuted (permutation seed = sample index), un-permuted; max |diff|. Same
    noise, frame rotation off, bf16 as in training.
  Z6 (spec 15.2, part of `suites`): order-free set error = (mean over predicted points of the
    distance to the nearest true point + mean over true points of the distance to the nearest
    predicted point) / 2, best of K; reported as `set_err_best_median` next to the ordered
    `err_best_median`, for the raw and the smoothed VA samples and for the VLA.
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
from examples.libero_r1 import diag_r1y as Y
from examples.libero_r1 import traj_select as TS

SUITE_CLS = {"libero_goal": "LiberoGoal", "libero_spatial": "LiberoSpatial",
             "libero_object": "LiberoObject", "libero_10": "Libero10"}
Z2_TIMESTEPS = (0, 1, 2, 5, 10, 20, 40, 60, 80, 99)
N_NOISE = 4
STEP_IDS = X2.STEP_IDS


def smooth_positions(pos: np.ndarray, window: int = 7) -> np.ndarray:
    """Centred moving average along time for (K, T, 3); window truncated at the ends."""
    K, T, _ = pos.shape
    h = window // 2
    out = np.empty_like(pos)
    for t in range(T):
        lo, hi = max(0, t - h), min(T, t + h + 1)
        out[:, t] = pos[:, lo:hi].mean(axis=1)
    return out


def gt_arrays(rec):
    fut = rec["gt_future_ee_states"]
    return (fut[:, 0, :16].reshape(-1, 4, 4)[:, :3, 3].double().numpy(), fut[:, 0, 16].double().numpy())


def set_error(ppos, gpos):
    """(K,T,3) predicted vs (T,3) true -> (K,) symmetric mean nearest-point distance."""
    d = np.linalg.norm(ppos[:, :, None, :] - gpos[None, None, :, :], axis=-1)   # (K, T, T)
    return 0.5 * (d.min(axis=2).mean(axis=1) + d.min(axis=1).mean(axis=1))


def traj_metrics(ppos, pgrip, rec, with_clusters):
    gpos, _ = gt_arrays(rec)
    adj = np.linalg.norm(np.diff(ppos, axis=1), axis=-1)       # (K, T-1)
    sg = Y.step_and_grip(ppos, pgrip, rec["gt_future_ee_states"])
    out = dict(adj=adj.ravel(), adj_sample_median=float(np.median(adj)), err_best=sg["err_best"],
               set_err_best=float(set_error(ppos, gpos).min()),
               **{f"err_step{k}": sg[f"err_step{k}"] for k in STEP_IDS})
    out["n_clusters"] = TS.make_candidates(ppos, pgrip).n_clusters if with_clusters else 1
    return out


def summarize(recs):
    pooled = np.concatenate([r["adj"] for r in recs])
    s = dict(n=len(recs), adj_median_pooled=float(np.median(pooled)), adj_p95_pooled=float(np.percentile(pooled, 95)),
             adj_median_of_samples=float(np.median([r["adj_sample_median"] for r in recs])),
             err_best_median=float(np.median([r["err_best"] for r in recs])),
             set_err_best_median=float(np.median([r["set_err_best"] for r in recs])),
             n_clusters_mean=float(np.mean([r["n_clusters"] for r in recs])))
    s.update({f"err_step{k}_median": float(np.median([r[f"err_step{k}"] for r in recs])) for k in STEP_IDS})
    return s


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def flat_row(tag, i, r):
    return dict(tag=tag, sample=i, adj_sample_median=r["adj_sample_median"], err_best=r["err_best"],
                set_err_best=r["set_err_best"],
                n_clusters=r["n_clusters"], **{f"err_step{k}": r[f"err_step{k}"] for k in STEP_IDS})


def cm(x):
    return f"{x * 100:.2f}"


# ── Z1 / Z3 / Z4 ─────────────────────────────────────────────────────────────

def run_suites(args, device):
    samples = {s: X2.build_samples(os.path.join(args.out_dir, f"samples_{s}.pt"), SUITE_CLS[s], args.n_samples)
               for s in args.suites}
    for s, smp in samples.items():
        with h5py.File(smp[0]["_episode_file"], "r") as h5:
            assert "ee_pose_desired" in h5, f"{s}: samples are not from the new (desired-pose) data; check APT_DATA_LOC"
    res, rows = {s: {} for s in args.suites}, []
    # ground truth (model independent)
    for s, smp in samples.items():
        gt = []
        for rec in smp:
            gpos, _ = gt_arrays(rec)
            a = np.linalg.norm(np.diff(gpos, axis=0), axis=-1)
            gt.append(a)
        pooled = np.concatenate(gt)
        res[s]["ground_truth"] = dict(adj_median_pooled=float(np.median(pooled)),
                                      adj_p95_pooled=float(np.percentile(pooled, 95)),
                                      adj_median_of_samples=float(np.median([np.median(a) for a in gt])))
    for label, ckpt in (("va", args.va), ("vla", args.vla)):
        model = Y.load(ckpt, device)
        stage = model.r1_meta["train_stage"]
        K = X2.K_VA if stage == 0 else 1
        for s, smp in samples.items():
            raw, smo = [], []
            for i, rec in enumerate(smp):
                d = X2.batch1(rec, device)
                ppos, pgrip = X2.predict(model, d, K, seed=i)
                r = traj_metrics(ppos, pgrip, rec, with_clusters=(stage == 0))
                raw.append(r)
                rows.append(dict(suite=s, model=label, **flat_row("raw", i, r)))
                if stage == 0:
                    r2 = traj_metrics(smooth_positions(ppos), pgrip, rec, with_clusters=True)
                    smo.append(r2)
                    rows.append(dict(suite=s, model=label, **flat_row("smooth7", i, r2)))
            res[s][label] = summarize(raw)
            res[s][label].update(ckpt=ckpt, iter=model.r1_meta["current_iters"], K=K)
            if smo:
                res[s][label + "_smooth7"] = summarize(smo)
            g = res[s]["ground_truth"]
            line = (f"[Z] {s:<14} {label:<4} adj med/p95 {cm(res[s][label]['adj_median_pooled'])}/"
                    f"{cm(res[s][label]['adj_p95_pooled'])} cm (GT {cm(g['adj_median_pooled'])}/{cm(g['adj_p95_pooled'])}), "
                    f"best err {cm(res[s][label]['err_best_median'])} cm (set {cm(res[s][label]['set_err_best_median'])}), "
                    f"step1 {cm(res[s][label]['err_step1_median'])} cm, "
                    f"clusters {res[s][label]['n_clusters_mean']:.1f}")
            if smo:
                sm = res[s][label + "_smooth7"]
                line += (f" | smooth7: adj {cm(sm['adj_median_pooled'])} cm, best err {cm(sm['err_best_median'])} cm "
                         f"(set {cm(sm['set_err_best_median'])}), "
                         f"step1 {cm(sm['err_step1_median'])} cm, clusters {sm['n_clusters_mean']:.1f}")
            print(line, flush=True)
        del model
        torch.cuda.empty_cache()
    write_rows(os.path.join(args.out_dir, "z_suites_rows.csv"), rows)
    with open(os.path.join(args.out_dir, "z_suites_summary.json"), "w") as fp:
        json.dump(res, fp, indent=2)
    if "libero_goal" in res and "vla" in res["libero_goal"]:
        e = res["libero_goal"]["vla"]["err_best_median"]
        print(f"[Z-SMOKE-2] Z1 VLA best-trajectory error median (LiberoGoal) = {cm(e)} cm (<= 3 cm) -> "
              f"{'PASS' if e <= 0.03 else 'FAIL'}")


# ── Z2 ───────────────────────────────────────────────────────────────────────

def fixed_t_table(model, samples, device, timesteps=Z2_TIMESTEPS):
    from apt.infer.planner import _repeat_batch
    acc = {t: {"pos_loss": [], "x0_err": []} for t in timesteps}
    for i, rec in enumerate(samples):
        d = X2.batch1(rec, device)
        gpos, _ = gt_arrays(rec)
        with torch.inference_mode():
            vl_in, vl_out = model.vlm(obs_rgbs=d["obs_rgbs"], obs_masks=d["obs_masks"],
                                      obs_norm_xys=d["obs_norm_xys"], obs_extrinsics=d["obs_extrinsics"],
                                      prompt_text=d["prompt_text"], fp16=True)
            for t in timesteps:
                torch.manual_seed(1000 * i + t)
                r = _repeat_batch(d, N_NOISE)
                _l, met, x0 = model.actor(
                    vl_inputs=_repeat_batch(vl_in, N_NOISE), vl_outputs=_repeat_batch(vl_out, N_NOISE),
                    current_ee_pose=r["current_ee_pose"], action_ref_pose=r["action_ref_pose"],
                    history_ee_states=r["history_ee_states"], gt_future_ee_states=r["gt_future_ee_states"],
                    valid_ee_mask=r["valid_ee_mask"], inference=False, fp16=True,
                    train_timesteps=torch.full((N_NOISE,), t, dtype=torch.long, device=d["obs_rgbs"].device),
                    return_x0=True)
                xpos = x0.float()[:, :, 0, :16].reshape(N_NOISE, -1, 4, 4)[..., :3, 3].double().cpu().numpy()
                acc[t]["pos_loss"].append(float(met["pos_loss"]))
                acc[t]["x0_err"].append(float(np.linalg.norm(xpos - gpos[None], axis=-1).mean()))
    return {str(t): {"pos_loss": float(np.mean(v["pos_loss"])), "x0_pos_err_m": float(np.mean(v["x0_err"]))}
            for t, v in acc.items()}


def run_timesteps(args, device):
    samples = X2.build_samples(os.path.join(args.out_dir, "samples_libero_goal.pt"), "LiberoGoal", args.n_samples)
    res = {}
    for label, ckpt in (("va", args.va), ("vla", args.vla)):
        model = Y.load(ckpt, device)
        res[label] = dict(ckpt=ckpt, iter=model.r1_meta["current_iters"], by_t=fixed_t_table(model, samples, device))
        print(f"[Z2] {label}: " + ", ".join(f"t{t}: loss {v['pos_loss']:.4f} / x0 {cm(v['x0_pos_err_m'])} cm"
                                            for t, v in res[label]["by_t"].items()), flush=True)
        del model
        torch.cuda.empty_cache()
    with open(os.path.join(args.out_dir, "z2_summary.json"), "w") as fp:
        json.dump(res, fp, indent=2)


# ── 13.3 overfit post-training check ─────────────────────────────────────────

def load_with_training_vlm(ckpt, device, cache):
    """Load a checkpoint for inference WITH the VLM weights it was trained with.

    A frozen-VLM run initialised from a checkpoint that carries `vlm_weights` (e.g. Q from
    apt_vla, whose VLM was fully fine-tuned) trains on top of those VLM weights, but its own
    checkpoints do not store them (the trainer saves VLM weights only when it trains the VLM),
    so the plain loader would fall back to the base Qwen3-VL. Here the run's TrainConfig JSON
    is read, and when the checkpoint has no `vlm_weights` but its `pretrained_ckpt` does, those
    are loaded into the VLM (unexpected keys are an error). Returns (model, vlm_source)."""
    import glob as _glob
    model = Y.load(ckpt, device)
    if model.r1_meta.get("has_vlm_weights"):
        return model, ckpt
    cfgs = sorted(_glob.glob(os.path.join(os.path.dirname(ckpt), "*.json")))
    pre = json.load(open(cfgs[-1])).get("pretrained_ckpt") if cfgs else None
    if not pre:
        return model, "base Qwen3-VL"
    if pre not in cache:
        src = torch.load(pre, map_location="cpu", weights_only=False)
        cache[pre] = src.get("vlm_weights")
        del src
    if cache[pre] is None:
        return model, "base Qwen3-VL"
    incompat = model.vlm.load_state_dict(cache[pre], strict=False)
    assert not incompat.unexpected_keys, f"unexpected VLM keys from {pre}: {incompat.unexpected_keys[:5]}"
    print(f"[OVERFIT] {ckpt}: VLM weights taken from its pretrained_ckpt {pre} "
          f"(missing={len(incompat.missing_keys)}, unexpected=0)")
    return model, pre


def run_overfit(args, device):
    samples = X2.build_samples(os.path.join(args.out_dir, "samples_goal_task1.pt"), "LiberoGoalTask1", args.n_samples)
    res, rows, vlm_cache = {}, [], {}
    for item in args.models:
        label, ckpt = item.split("=", 1)
        model, vlm_src = load_with_training_vlm(ckpt, device, vlm_cache)
        stage = model.r1_meta["train_stage"]
        recs = []
        for i, rec in enumerate(samples):
            d = X2.batch1(rec, device)   # prompt = the sample's true instruction (used by Stage-1 only)
            ppos, pgrip = X2.predict(model, d, X2.K_VA, seed=i)
            r = traj_metrics(ppos, pgrip, rec, with_clusters=True)
            recs.append(r)
            rows.append(dict(model=label, **flat_row("raw", i, r)))
        s = summarize(recs)
        s.update(ckpt=ckpt, iter=model.r1_meta["current_iters"], train_stage=stage, vlm_source=vlm_src,
                 by_t=fixed_t_table(model, samples, device))
        res[label] = s
        print(f"[OVERFIT] {label} (stage {stage}, iter {s['iter']}): adj median {cm(s['adj_median_pooled'])} cm, "
              f"best err {cm(s['err_best_median'])} cm, step1 {cm(s['err_step1_median'])} cm, x0 err t0/t10/t40 = "
              f"{cm(s['by_t']['0']['x0_pos_err_m'])}/{cm(s['by_t']['10']['x0_pos_err_m'])}/"
              f"{cm(s['by_t']['40']['x0_pos_err_m'])} cm", flush=True)
        del model
        torch.cuda.empty_cache()
    write_rows(os.path.join(args.out_dir, "overfit_rows.csv"), rows)
    with open(os.path.join(args.out_dir, "overfit_summary.json"), "w") as fp:
        json.dump(res, fp, indent=2)


# ── Z5 permutation equivariance ──────────────────────────────────────────────

Z5_TIMESTEPS = (5, 50, 95)


def run_equivariance(args, device):
    samples = X2.build_samples(os.path.join(args.out_dir, "samples_goal20.pt"), "LiberoGoal", args.n_samples)
    res, rows = {}, []
    for label, ckpt in (("va", args.va), ("vla", args.vla)):
        model = Y.load(ckpt, device)
        act_dim = model.actor.act_dim
        diffs = {t: [] for t in Z5_TIMESTEPS}
        for i, rec in enumerate(samples):
            d = X2.batch1(rec, device)
            Ta, nee = d["gt_future_ee_states"].shape[1:3]
            noise = torch.from_numpy(np.random.default_rng(i).standard_normal((1, nee, Ta, act_dim)).astype(np.float32)).to(device)
            perm = torch.from_numpy(np.random.default_rng(i).permutation(Ta)).to(device)
            inv = torch.argsort(perm)
            with torch.inference_mode():
                vl_in, vl_out = model.vlm(obs_rgbs=d["obs_rgbs"], obs_masks=d["obs_masks"],
                                          obs_norm_xys=d["obs_norm_xys"], obs_extrinsics=d["obs_extrinsics"],
                                          prompt_text=d["prompt_text"], fp16=True)
                for t in Z5_TIMESTEPS:
                    outs = []
                    for p in (None, perm):
                        gt = d["gt_future_ee_states"] if p is None else d["gt_future_ee_states"][:, p]
                        nz = noise if p is None else noise[:, :, p]
                        _l, _m, e = model.actor(
                            vl_inputs=dict(vl_in), vl_outputs=dict(vl_out), current_ee_pose=d["current_ee_pose"],
                            action_ref_pose=d["action_ref_pose"], history_ee_states=d["history_ee_states"],
                            gt_future_ee_states=gt, valid_ee_mask=d["valid_ee_mask"], inference=False, fp16=True,
                            train_timesteps=torch.full((1,), t, dtype=torch.long, device=device), train_noise=nz,
                            random_frame_rotation=False, return_eps=True)
                        outs.append(e.float() if p is None else e.float()[:, :, inv])
                    dd = float((outs[0] - outs[1]).abs().max())
                    diffs[t].append(dd)
                    rows.append(dict(model=label, sample=i, t=t, max_abs_diff=dd))
        res[label] = dict(ckpt=ckpt, iter=model.r1_meta["current_iters"],
                          stage0_action_rope=model.r1_meta.get("stage0_action_rope", "off"),
                          by_t={str(t): dict(max=float(np.max(v)), median=float(np.median(v))) for t, v in diffs.items()})
        print(f"[Z5] {label}: " + ", ".join(f"t{t}: max {np.max(v):.3e} / median {np.median(v):.3e}"
                                            for t, v in diffs.items()), flush=True)
        del model
        torch.cuda.empty_cache()
    va_max = max(v["max"] for v in res["va"]["by_t"].values())
    vla_min = min(v["median"] for v in res["vla"]["by_t"].values())
    res["verdict"] = dict(va_equivariant=bool(va_max <= 1e-2), vla_not_equivariant=bool(vla_min > 0.1))
    write_rows(os.path.join(args.out_dir, "z5_rows.csv"), rows)
    with open(os.path.join(args.out_dir, "z5_summary.json"), "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"[Z5-VERDICT] VA max diff {va_max:.3e} (<= 1e-2: equivariant={va_max <= 1e-2}); "
          f"VLA smallest median diff {vla_min:.3e} (> 0.1: not equivariant={vla_min > 0.1})")


# ── 17.3 R1-T3 post-training check and checkpoint choice ─────────────────────

R1C_ADJ_MAX = 0.015        # criterion 1: pooled adjacent-step median, every suite
R1C_STEP1_MAX = 0.02       # criterion 2: step-1 error median, every suite
R1C_ERR_MAX = 0.03         # criterion 3: best-of-8 error median, Spatial and Object only
R1C_ERR_SUITES = ("libero_spatial", "libero_object")


def run_r1c_check(args, device):
    """Spec 17.3 criteria 1-3 for every --models checkpoint (Z1/Z3/Z6 metrics, 4 suites x 200
    samples, 8 trajectories each); criterion 4 (smoke 11) is added by `r1c_select`."""
    samples = {s: X2.build_samples(os.path.join(args.out_dir, f"samples_{s}.pt"), SUITE_CLS[s], args.n_samples)
               for s in args.suites}
    for s, smp in samples.items():
        with h5py.File(smp[0]["_episode_file"], "r") as h5:
            assert "ee_pose_desired" in h5, f"{s}: samples are not from the new (desired-pose) data; check APT_DATA_LOC"
    res, rows = {"ground_truth": {}}, []
    for s, smp in samples.items():
        pooled = np.concatenate([np.linalg.norm(np.diff(gt_arrays(rec)[0], axis=0), axis=-1) for rec in smp])
        res["ground_truth"][s] = dict(adj_median_pooled=float(np.median(pooled)), adj_p95_pooled=float(np.percentile(pooled, 95)))
    models = [item.split("=", 1) for item in args.models] + ([["vla", args.vla]] if args.with_vla else [])
    for label, ckpt in models:
        model = Y.load(ckpt, device)
        stage = model.r1_meta["train_stage"]
        K = X2.K_VA if stage == 0 else 1
        res[label] = dict(ckpt=ckpt, iter=model.r1_meta["current_iters"], train_stage=stage,
                          stage0_action_rope=model.r1_meta.get("stage0_action_rope", "off"), suites={})
        for s, smp in samples.items():
            recs = []
            for i, rec in enumerate(smp):
                d = X2.batch1(rec, device)
                ppos, pgrip = X2.predict(model, d, K, seed=i)
                r = traj_metrics(ppos, pgrip, rec, with_clusters=(stage == 0))
                recs.append(r)
                rows.append(dict(suite=s, model=label, **flat_row("raw", i, r)))
            res[label]["suites"][s] = summarize(recs)
            m = res[label]["suites"][s]
            print(f"[R1C] {label:<6} {s:<14} adj {cm(m['adj_median_pooled'])} cm, step1 {cm(m['err_step1_median'])} cm, "
                  f"best err {cm(m['err_best_median'])} cm, set err {cm(m['set_err_best_median'])} cm, "
                  f"clusters {m['n_clusters_mean']:.1f}", flush=True)
        if stage == 0:
            su = res[label]["suites"]
            c1 = all(su[s]["adj_median_pooled"] <= R1C_ADJ_MAX for s in su)
            c2 = all(su[s]["err_step1_median"] <= R1C_STEP1_MAX for s in su)
            c3 = all(su[s]["err_best_median"] <= R1C_ERR_MAX for s in R1C_ERR_SUITES if s in su)
            res[label]["criteria"] = dict(c1_adjacent=bool(c1), c2_step1=bool(c2), c3_err_spatial_object=bool(c3),
                                          c1_to_c3=bool(c1 and c2 and c3 and set(R1C_ERR_SUITES) <= set(su)))
            print(f"[R1C] {label}: criteria 1/2/3 = {c1}/{c2}/{c3}", flush=True)
        del model
        torch.cuda.empty_cache()
    write_rows(os.path.join(args.out_dir, "r1c_check_rows.csv"), rows)
    with open(os.path.join(args.out_dir, "r1c_check_summary.json"), "w") as fp:
        json.dump(res, fp, indent=2)


def _smoke11_passed(log_path, kind):
    """kind 'aligned': check_sampling --smoke11 log; 'fp32': --smoke11_unpadded on an fp32 service."""
    if not log_path or not os.path.exists(log_path):
        return False
    text = open(log_path, encoding="utf-8", errors="replace").read()
    if kind == "aligned":
        return "[SMOKE-11] overall (VA side): PASS" in text
    return "[SMOKE-11-UNPADDED] fp32" in text and "-> PASS" in text.split("[SMOKE-11-UNPADDED] fp32", 1)[1]


def run_r1c_select(args, device=None):
    """Spec 17.3: 40k if it passes criteria 1-4, else 30k, else stop. --smoke11 entries are
    label=<aligned bf16 log>,<fp32 unpadded log>."""
    with open(args.check_json) as fp:
        chk = json.load(fp)
    logs = {}
    for item in args.smoke11:
        label, paths = item.split("=", 1)
        a, b = (paths.split(",") + [""])[:2]
        logs[label] = dict(aligned=_smoke11_passed(a, "aligned"), fp32=_smoke11_passed(b, "fp32"), logs=[a, b])
    decision = {"order": args.prefer, "candidates": {}}
    chosen = None
    for label in args.prefer:
        c = chk.get(label, {}).get("criteria", {})
        sm = logs.get(label, {"aligned": False, "fp32": False, "logs": []})
        ok = bool(c.get("c1_to_c3")) and sm["aligned"] and sm["fp32"]
        decision["candidates"][label] = dict(criteria_1_3=c, smoke11_aligned=sm["aligned"], smoke11_fp32=sm["fp32"],
                                             smoke11_logs=sm["logs"], passed=ok)
        print(f"[R1C-SELECT] {label}: criteria 1-3 {c.get('c1_to_c3')}, smoke 11 aligned {sm['aligned']}, "
              f"fp32 unpadded {sm['fp32']} -> {'PASS' if ok else 'FAIL'}")
        if ok and chosen is None:
            chosen = label
    decision["selected"] = None if chosen is None else dict(label=chosen, ckpt=chk[chosen]["ckpt"], iter=chk[chosen]["iter"])
    out_dir = os.path.dirname(os.path.abspath(args.check_json))
    with open(os.path.join(out_dir, "r1c_selection.json"), "w") as fp:
        json.dump(decision, fp, indent=2)
    if chosen is None:
        print("[R1C-SELECT] no checkpoint passes -> STOP and report")
        raise SystemExit(1)
    with open(os.path.join(out_dir, "selected_ckpt.txt"), "w") as fp:
        fp.write(os.path.abspath(chk[chosen]["ckpt"]) + "\n")
    print(f"[R1C-SELECT] use {chosen}: {chk[chosen]['ckpt']} (iter {chk[chosen]['iter']})")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda:0"
    if args.cmd == "r1c_select":
        return run_r1c_select(args)
    {"suites": run_suites, "timesteps": run_timesteps, "overfit": run_overfit,
     "equivariance": run_equivariance, "r1c_check": run_r1c_check}[args.cmd](args, device)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["suites", "timesteps", "overfit", "equivariance", "r1c_check", "r1c_select"])
    p.add_argument("--va", default="checkpoints/APT/va_libero4_r1b/ckpt_0060000.pt")
    p.add_argument("--vla", default="checkpoints/hf/apt_vla_ftlibero/ckpt_latest.pt")
    p.add_argument("--models", nargs="*", default=[], help="overfit / r1c_check: label=ckpt")
    p.add_argument("--with_vla", action="store_true", help="r1c_check: also record the VLA reference (Z1)")
    p.add_argument("--check_json", default="results/R1/R1T3/check/r1c_check_summary.json", help="r1c_select")
    p.add_argument("--smoke11", nargs="*", default=[], help="r1c_select: label=<aligned bf16 log>,<fp32 unpadded log>")
    p.add_argument("--prefer", nargs="*", default=["va40k", "va30k"], help="r1c_select order (spec 17.3)")
    p.add_argument("--suites", nargs="+", default=list(SUITE_CLS), choices=list(SUITE_CLS))
    p.add_argument("--n_samples", type=int, default=None,
                   help="default 200 (suites, timesteps) / 100 (overfit) / 20 (equivariance)")
    p.add_argument("--out_dir", default="results/R1/R1Z")
    a = p.parse_args()
    if a.n_samples is None:
        a.n_samples = {"overfit": 100, "equivariance": 20}.get(a.cmd, 200)
    main(a)
