"""R1-X / X2: open-loop check on the converted training data (spec section 9.2).
APT env, repo root, one GPU, APT_DATA_LOC_HOST set. No inference server needed.

  CUDA_VISIBLE_DEVICES=0 python -m examples.libero_r1.diag_r1x_x2 \
      --models apt_va=checkpoints/hf/apt_va/ckpt_latest.pt \
               va40k=checkpoints/APT/va_libero4_r1/ckpt_0040000.pt \
               vla=checkpoints/hf/apt_vla_ftlibero/ckpt_latest.pt \
      --out_dir results/R1/R1X

Data: `LiberoGoal` read through the training loader (`H5DatasetMapBase.__getitem__`)
with every augmentation switched off (photometric, noise, geometric, background,
camera shuffling). The reference camera is still drawn per sample by the loader's
rule (uniform over agentview / eye_in_hand), seeded. 200 samples are drawn once
with seed 0 and cached (x2_samples.pt) so every model sees the same samples.

Per sample and model:
  predictions  VA: 8 trajectories, VLA: 1 (noise = default_rng(sample index));
               32-step mean position error to the ground truth (VA: min and mean
               over the 8), predicted adjacent-step displacement median.
  loss         training loss (pos / rot / openness / total) in 5 diffusion
               timestep bins [0-19, 20-39, 40-59, 60-79, 80-99], 4 random
               timesteps per bin per sample (seeded), computed exactly like
               training (ActionExpert training branch, fp16 autocast).
Ground-truth statistics (adjacent-step displacement median / p95 / max, net
displacement) are model independent.
Outputs: x2_gt.csv, x2_pred.csv, x2_loss.csv, x2_summary.json. Smoke check 3
(spec section 11): VLA median position error <= 4 cm and net-displacement ratio
pred/gt in 0.7-1.4. Spec 11.3 extra records (X2 and X2b): position error at
future steps 1, 2, 4, 8, 16, 32 and gripper |pred - gt| on the trajectory with the
smallest mean position error (VA: best of 8), plus mean gt / pred gripper openness.

X2b (spec section 10, ruling 4) -- inference-setting control on the SAME cached
samples and noise seeds, only for the VA models passed in --models:

  CUDA_VISIBLE_DEVICES=0 python -m examples.libero_r1.diag_r1x_x2 --x2b \
      --models apt_va=checkpoints/hf/apt_va/ckpt_latest.pt \
               va40k=checkpoints/APT/va_libero4_r1/ckpt_0040000.pt \
      --out_dir results/R1/R1X

  diffusion inference steps {20, 100} x action-expert precision {bf16, fp32},
  8 trajectories per sample. Steps are set on the loaded model instance only
  (ActionExpert.inference_timesteps); precision is the action expert's `fp16`
  flag. The frozen VLM always runs in bf16 as in training: its weights are
  loaded in bf16 and CUDA autocast has no fp32 mode, so for fp32 the VLM
  features are cast to fp32 before the action expert. No default path changes.
  Outputs x2b_pred.csv (one row per model x setting x sample) and x2b_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

import numpy as np
import torch

N_SAMPLES = 200
X2B_STEPS = (20, 100)
X2B_PRECISIONS = ("bf16", "fp32")
SEED = 0
K_VA = 8
BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
T_PER_BIN = 4


def motion(pos):
    from examples.libero_r1.traj_select import motion_stats
    return motion_stats(pos, np.ones(pos.shape[:2]))


def build_samples(path, cls_name="LiberoGoal", n=N_SAMPLES):
    """n samples (seed 0) through the training loader of dataset class `cls_name`, all
    augmentation off; cached at `path`. Defaults reproduce X2 exactly."""
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    from data_utils import datasets
    D = getattr(datasets, cls_name)
    cfg = D.config
    cfg.photometric_aug = False
    cfg.noise_aug = False
    cfg.geom_aug = False
    cfg.background_aug = False
    cfg.shuffle_cameras = False
    ds = D.inst(train_stage=0)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    idx = np.random.default_rng(SEED).choice(len(ds), size=n, replace=len(ds) < n)
    samples = []
    for i, ep in enumerate(idx):
        x = ds[int(ep)]
        rec = {k: (torch.as_tensor(np.asarray(v)) if not isinstance(v, str) else v)
               for k, v in x.items() if v is not None}
        rec["_episode_index"] = int(ep)
        rec["_episode_file"] = ds.h5_filelist[int(ep)]
        samples.append(rec)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(samples, path)
    print(f"[X2] cached {len(samples)} samples from {len(ds)} {cls_name} episodes -> {path}")
    return samples


def batch1(rec, device):
    out = {}
    for k in ("obs_rgbs", "obs_masks", "obs_norm_xys", "obs_extrinsics", "current_ee_pose", "action_ref_pose",
              "history_ee_states", "gt_future_ee_states", "valid_ee_mask"):
        out[k] = rec[k].unsqueeze(0).to(device) if k in rec else None
    out["prompt_text"] = [rec["prompt_text"]]
    return out


def _to_float32(obj):
    if isinstance(obj, torch.Tensor):
        return obj.float() if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: _to_float32(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_float32(v) for v in obj]
    return obj


def predict(model, d, Kp, seed, steps=None, fp16=True):
    """Kp trajectories for one batch-1 sample; noise = default_rng(seed). Returns
    predicted positions (Kp, Ta, 3) and grippers (Kp, Ta) as float64 numpy.
    steps / fp16 only change this call (restored afterwards)."""
    from apt.infer.planner import _repeat_batch
    act_dim = model.actor.act_dim
    old_steps = model.actor.inference_timesteps
    if steps is not None:
        model.actor.inference_timesteps = int(steps)
    try:
        with torch.inference_mode():
            vl_in, vl_out = model.vlm(obs_rgbs=d["obs_rgbs"], obs_masks=d["obs_masks"],
                                      obs_norm_xys=d["obs_norm_xys"], obs_extrinsics=d["obs_extrinsics"],
                                      prompt_text=d["prompt_text"], fp16=True)  # VLM always bf16 (as trained)
            if not fp16:
                vl_in, vl_out = _to_float32(vl_in), _to_float32(vl_out)
            Ta, nee = d["gt_future_ee_states"].shape[1:3]
            noise = torch.from_numpy(np.random.default_rng(seed).standard_normal((Kp, nee, Ta, act_dim))
                                     .astype(np.float32)).to(d["obs_rgbs"].device)
            rep = _repeat_batch(d, Kp)
            pred = model.actor(vl_inputs=_repeat_batch(vl_in, Kp), vl_outputs=_repeat_batch(vl_out, Kp),
                               current_ee_pose=rep["current_ee_pose"], action_ref_pose=rep["action_ref_pose"],
                               history_ee_states=rep["history_ee_states"],
                               gt_future_ee_states=rep["gt_future_ee_states"], valid_ee_mask=rep["valid_ee_mask"],
                               inference=True, fp16=fp16, initial_noise=noise)
    finally:
        model.actor.inference_timesteps = old_steps
    pred = pred.float()[:, :, 0]
    ppos = pred[..., :16].reshape(Kp, Ta, 4, 4)[..., :3, 3].double().cpu().numpy()
    pgrip = pred[..., 16].double().cpu().numpy()
    return ppos, pgrip


def gt_pos(rec):
    return rec["gt_future_ee_states"][:, 0, :16].reshape(-1, 4, 4)[:, :3, 3].double().numpy()


def gt_grip(rec):
    return rec["gt_future_ee_states"][:, 0, 16].double().numpy()


def cur_pos(rec):
    return rec["current_ee_pose"][0, :3, 3].double().numpy()


STEP_IDS = (1, 2, 4, 8, 16, 32)  # spec 11.3, 1-based future steps


def sample_record(ppos, pgrip, rec):
    """Per-sample comparison of Kp predictions with the ground truth (spec 9.2, 11.3).
    The 'best' trajectory is the one with the smallest 32-step mean position error
    (for the VLA, Kp = 1, it is the only one); step errors and gripper errors are
    taken on it. Net displacements are measured from the current EE position for
    both prediction and ground truth."""
    gt, gg, cur = gt_pos(rec), gt_grip(rec), cur_pos(rec)
    step_err = np.linalg.norm(ppos - gt[None], axis=-1)        # (Kp, Ta)
    e = step_err.mean(axis=1)                                 # (Kp,)
    b = int(np.argmin(e))
    pred_net = np.linalg.norm(ppos[:, -1] - cur[None], axis=-1)
    rec_out = dict(err_min=float(e.min()), err_mean=float(e.mean()),
                   pred_net_from_cur_median=float(np.median(pred_net)),
                   gt_net_from_cur=float(np.linalg.norm(gt[-1] - cur)),
                   grip_mae_best=float(np.abs(pgrip[b] - gg).mean()),
                   grip_gt_mean=float(gg.mean()), grip_pred_mean_best=float(pgrip[b].mean()))
    for k in STEP_IDS:
        rec_out[f"err_step{k}_best"] = float(step_err[b, k - 1])
    return rec_out


EXTRA_FIELDS = (["pred_net_from_cur_median", "gt_net_from_cur", "grip_mae_best", "grip_gt_mean",
                 "grip_pred_mean_best"] + [f"err_step{k}_best" for k in STEP_IDS])


def extra_summary(rows):
    """Spec 11.3 aggregates over samples."""
    out = {f"err_step{k}_median": float(np.median([r[f"err_step{k}_best"] for r in rows])) for k in STEP_IDS}
    out.update(grip_mae_median=float(np.median([r["grip_mae_best"] for r in rows])),
               grip_gt_mean=float(np.mean([r["grip_gt_mean"] for r in rows])),
               grip_pred_mean=float(np.mean([r["grip_pred_mean_best"] for r in rows])),
               pos_err_best_median=float(np.median([r["err_min"] for r in rows])),
               pred_net_median=float(np.median([r["pred_net_from_cur_median"] for r in rows])),
               gt_net_median=float(np.median([r["gt_net_from_cur"] for r in rows])))
    out["net_ratio_pred_over_gt"] = out["pred_net_median"] / max(out["gt_net_median"], 1e-12)
    return out


def run_x2b(label, ckpt, samples, device, writer):
    from apt.infer.planner import load_model
    from examples.libero_r1.traj_select import motion_stats
    model, _cfg = load_model(ckpt, device)
    assert model.r1_meta["train_stage"] == 0, f"X2b is for VA models only: {label}"
    out = {}
    for steps in X2B_STEPS:
        for prec in X2B_PRECISIONS:
            acc = []
            for i, rec in enumerate(samples):
                d = batch1(rec, device)
                ppos, pgrip = predict(model, d, K_VA, seed=i, steps=steps, fp16=(prec == "bf16"))
                m = motion_stats(ppos, pgrip, cur_pos=cur_pos(rec))
                sr = sample_record(ppos, pgrip, rec)
                row = dict(model=label, steps=steps, precision=prec, sample=i,
                           adj_disp_median=round(m["adj_disp_median"], 6),
                           p95_over_median=round(m["adj_disp_p95"] / max(m["adj_disp_median"], 1e-12), 4),
                           net_disp_median=round(m["net_disp_median"], 6),
                           grip_std_median=round(m["grip_std_median"], 6),
                           **{k: round(v, 6) for k, v in sr.items()})
                writer.writerow(row)
                acc.append(row)
            key = f"steps{steps}_{prec}"
            out[key] = {k: float(np.median([r[k] for r in acc])) for k in
                        ("adj_disp_median", "p95_over_median", "net_disp_median", "grip_std_median")}
            out[key].update(err_min_mean=float(np.mean([r["err_min"] for r in acc])),
                            err_mean_mean=float(np.mean([r["err_mean"] for r in acc])), n=len(acc))
            out[key].update(extra_summary(acc))
            print(f"[X2b] {label} {key}: {out[key]}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def run_model(label, ckpt, samples, device, writers):
    from apt.infer.planner import _repeat_batch, load_model
    model, _cfg = load_model(ckpt, device)
    stage = model.r1_meta["train_stage"]
    errs, adj_pred, loss_acc, rows_extra = [], [], {b: [] for b in range(len(BINS))}, []
    for i, rec in enumerate(samples):
        d = batch1(rec, device)
        gt = gt_pos(rec)  # (32, 3)
        Kp = K_VA if stage == 0 else 1
        ppos, pgrip = predict(model, d, Kp, seed=i)  # default steps (20), bf16
        with torch.inference_mode():
            vl_in, vl_out = model.vlm(obs_rgbs=d["obs_rgbs"], obs_masks=d["obs_masks"],
                                      obs_norm_xys=d["obs_norm_xys"], obs_extrinsics=d["obs_extrinsics"],
                                      prompt_text=d["prompt_text"], fp16=True)
            e = np.linalg.norm(ppos - gt[None], axis=-1).mean(axis=1)  # (Kp,)
            m = motion(ppos)
            sr = sample_record(ppos, pgrip, rec)
            row = dict(model=label, train_stage=stage, sample=i, err_min=round(float(e.min()), 6),
                       err_mean=round(float(e.mean()), 6), pred_adj_disp_median=round(m["adj_disp_median"], 6),
                       pred_adj_disp_p95=round(m["adj_disp_p95"], 6), pred_net_disp_median=round(m["net_disp_median"], 6),
                       **{k: round(v, 6) for k, v in sr.items() if k not in ("err_min", "err_mean")})
            writers["pred"].writerow(row)
            rows_extra.append(sr)
            errs.append((e.min(), e.mean()))
            adj_pred.append(m["adj_disp_median"])

            for b, (lo, hi) in enumerate(BINS):
                g = torch.Generator().manual_seed(1000 * i + b)
                ts = torch.randint(lo, hi, (T_PER_BIN,), generator=g)
                torch.manual_seed(1000 * i + b)  # training noise + random frame rotation inside the actor
                r4 = _repeat_batch(d, T_PER_BIN)
                _loss, met = model.actor(vl_inputs=_repeat_batch(vl_in, T_PER_BIN),
                                         vl_outputs=_repeat_batch(vl_out, T_PER_BIN),
                                         current_ee_pose=r4["current_ee_pose"], action_ref_pose=r4["action_ref_pose"],
                                         history_ee_states=r4["history_ee_states"],
                                         gt_future_ee_states=r4["gt_future_ee_states"],
                                         valid_ee_mask=r4["valid_ee_mask"], inference=False, fp16=True,
                                         train_timesteps=ts.to(device))
                writers["loss"].writerow(dict(model=label, sample=i, bin=f"{lo}-{hi - 1}",
                                              **{k: round(float(v), 6) for k, v in met.items()}))
                loss_acc[b].append(met)
        if (i + 1) % 50 == 0:
            print(f"[X2] {label}: {i + 1}/{len(samples)}", flush=True)
    del model
    torch.cuda.empty_cache()
    errs = np.asarray(errs)
    summ = dict(train_stage=stage, ckpt=ckpt, n=len(samples),
                pos_err_mean_of_min=float(errs[:, 0].mean()), pos_err_mean_of_mean=float(errs[:, 1].mean()),
                pred_adj_disp_median=float(np.median(adj_pred)),
                loss_by_bin={f"{lo}-{hi - 1}": {k: float(np.mean([m[k] for m in loss_acc[b]]))
                                                 for k in ("pos_loss", "rot_loss", "openness_loss", "total_loss")}
                             for b, (lo, hi) in enumerate(BINS)})
    summ.update(extra_summary(rows_extra))
    return summ


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda:0"
    samples = build_samples(os.path.join(args.out_dir, "x2_samples.pt"))
    if args.x2b:
        res = {}
        with open(os.path.join(args.out_dir, "x2b_pred.csv"), "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=["model", "steps", "precision", "sample", "adj_disp_median",
                                               "p95_over_median", "net_disp_median", "err_min", "err_mean",
                                               "grip_std_median"] + EXTRA_FIELDS)
            w.writeheader()
            for item in args.models:
                label, ckpt = item.split("=", 1)
                res[label] = run_x2b(label, ckpt, samples, device, w)
                fp.flush()
        with open(os.path.join(args.out_dir, "x2b_summary.json"), "w") as fp:
            json.dump(res, fp, indent=2)
        return
    # ground truth statistics
    gts = []
    with open(os.path.join(args.out_dir, "x2_gt.csv"), "w", newline="") as fp:
        w = None
        for i, rec in enumerate(samples):
            gt = rec["gt_future_ee_states"][:, 0, :16].reshape(-1, 4, 4)[:, :3, 3].double().numpy()
            cur = rec["current_ee_pose"][0, :3, 3].double().numpy()
            st = motion(gt[None])
            st["net_disp_median"] = float(np.linalg.norm(gt[-1] - cur))
            row = dict(sample=i, episode_file=os.path.basename(rec["_episode_file"]), prompt=rec["prompt_text"],
                       **{k: round(v, 6) for k, v in st.items() if k != "grip_std_median"})
            if w is None:
                w = csv.DictWriter(fp, fieldnames=list(row.keys()))
                w.writeheader()
            w.writerow(row)
            gts.append(np.linalg.norm(np.diff(gt, axis=0), axis=-1))
            gts[-1] = (gts[-1], st["net_disp_median"])
    adj = np.concatenate([g[0] for g in gts])
    gt_summary = dict(adj_disp_median=float(np.median(adj)), adj_disp_p95=float(np.percentile(adj, 95)),
                      adj_disp_max=float(adj.max()), net_disp_median=float(np.median([g[1] for g in gts])))
    print(f"[X2] ground truth (m): {gt_summary}")

    summary = {"ground_truth": gt_summary, "models": {}}
    fp_pred = open(os.path.join(args.out_dir, "x2_pred.csv"), "w", newline="")
    fp_loss = open(os.path.join(args.out_dir, "x2_loss.csv"), "w", newline="")
    writers = {"pred": csv.DictWriter(fp_pred, fieldnames=["model", "train_stage", "sample", "err_min", "err_mean",
                                                           "pred_adj_disp_median", "pred_adj_disp_p95",
                                                           "pred_net_disp_median"] + EXTRA_FIELDS),
               "loss": csv.DictWriter(fp_loss, fieldnames=["model", "sample", "bin", "pos_loss", "rot_loss",
                                                           "openness_loss", "total_loss"])}
    writers["pred"].writeheader()
    writers["loss"].writeheader()
    try:
        for item in args.models:
            label, ckpt = item.split("=", 1)
            print(f"[X2] model {label}: {ckpt}")
            summary["models"][label] = run_model(label, ckpt, samples, device, writers)
            fp_pred.flush()
            fp_loss.flush()
            print(f"[X2] {label}: {json.dumps(summary['models'][label], indent=1)}", flush=True)
    finally:
        fp_pred.close()
        fp_loss.close()
    with open(os.path.join(args.out_dir, "x2_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    for label, s in summary["models"].items():
        if s["train_stage"] == 1:
            # spec section 11, ruling 1: median of the per-sample 32-step mean position error <= 4 cm
            # and median(pred net displacement) / median(gt net displacement) in [0.7, 1.4]
            e, r = s["pos_err_best_median"], s["net_ratio_pred_over_gt"]
            ok = e <= 0.04 and 0.7 <= r <= 1.4
            print(f"[X2-SMOKE-3] {label}: median 32-step position error = {e*100:.2f} cm (<= 4 cm), "
                  f"net displacement ratio pred/gt = {r:.2f} (0.7-1.4) -> "
                  f"{'PASS' if ok else 'FAIL (data reading/conversion differs from VLA training data)'}")
            print(f"[X2] {label}: step errors (m) " +
                  ", ".join(f"{k}:{s[f'err_step{k}_median']:.4f}" for k in STEP_IDS) +
                  f"; gripper MAE median {s['grip_mae_median']:.3f}, mean gt {s['grip_gt_mean']:.3f}, "
                  f"mean pred {s['grip_pred_mean']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="label=checkpoint path")
    p.add_argument("--out_dir", default="results/R1/R1X")
    p.add_argument("--x2b", action="store_true", help="run X2b (inference steps x precision) instead of X2")
    main(p.parse_args())
