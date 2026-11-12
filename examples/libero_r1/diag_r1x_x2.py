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
Outputs: x2_gt.csv, x2_pred.csv, x2_loss.csv, x2_summary.json; smoke check 3:
VLA mean position error <= 3 cm.
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
SEED = 0
K_VA = 8
BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
T_PER_BIN = 4


def motion(pos):
    from examples.libero_r1.traj_select import motion_stats
    return motion_stats(pos, np.ones(pos.shape[:2]))


def build_samples(path):
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    from data_utils import datasets
    cfg = datasets.LiberoGoal.config
    cfg.photometric_aug = False
    cfg.noise_aug = False
    cfg.geom_aug = False
    cfg.background_aug = False
    cfg.shuffle_cameras = False
    ds = datasets.LiberoGoal.inst(train_stage=0)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    idx = np.random.default_rng(SEED).choice(len(ds), size=N_SAMPLES, replace=len(ds) < N_SAMPLES)
    samples = []
    for i, ep in enumerate(idx):
        x = ds[int(ep)]
        rec = {k: (torch.as_tensor(np.asarray(v)) if not isinstance(v, str) else v)
               for k, v in x.items() if v is not None}
        rec["_episode_index"] = int(ep)
        rec["_episode_file"] = ds.h5_filelist[int(ep)]
        samples.append(rec)
    torch.save(samples, path)
    print(f"[X2] cached {len(samples)} samples from {len(ds)} LiberoGoal episodes -> {path}")
    return samples


def batch1(rec, device):
    out = {}
    for k in ("obs_rgbs", "obs_masks", "obs_norm_xys", "obs_extrinsics", "current_ee_pose", "action_ref_pose",
              "history_ee_states", "gt_future_ee_states", "valid_ee_mask"):
        out[k] = rec[k].unsqueeze(0).to(device) if k in rec else None
    out["prompt_text"] = [rec["prompt_text"]]
    return out


def run_model(label, ckpt, samples, device, writers):
    from apt.infer.planner import _repeat_batch, load_model
    model, _cfg = load_model(ckpt, device)
    stage = model.r1_meta["train_stage"]
    act_dim = model.actor.act_dim
    errs, adj_pred, loss_acc = [], [], {b: [] for b in range(len(BINS))}
    for i, rec in enumerate(samples):
        d = batch1(rec, device)
        gt = rec["gt_future_ee_states"][:, 0, :16].reshape(-1, 4, 4)[:, :3, 3].double().numpy()  # (32, 3)
        with torch.inference_mode():
            vl_in, vl_out = model.vlm(obs_rgbs=d["obs_rgbs"], obs_masks=d["obs_masks"],
                                      obs_norm_xys=d["obs_norm_xys"], obs_extrinsics=d["obs_extrinsics"],
                                      prompt_text=d["prompt_text"], fp16=True)
            Kp = K_VA if stage == 0 else 1
            Ta, nee = d["gt_future_ee_states"].shape[1:3]
            noise = torch.from_numpy(np.random.default_rng(i).standard_normal((Kp, nee, Ta, act_dim))
                                     .astype(np.float32)).to(device)
            rep = _repeat_batch(d, Kp)
            pred = model.actor(vl_inputs=_repeat_batch(vl_in, Kp), vl_outputs=_repeat_batch(vl_out, Kp),
                               current_ee_pose=rep["current_ee_pose"], action_ref_pose=rep["action_ref_pose"],
                               history_ee_states=rep["history_ee_states"],
                               gt_future_ee_states=rep["gt_future_ee_states"], valid_ee_mask=rep["valid_ee_mask"],
                               inference=True, fp16=True, initial_noise=noise)
            ppos = pred.float()[:, :, 0, :16].reshape(Kp, Ta, 4, 4)[..., :3, 3].double().cpu().numpy()
            e = np.linalg.norm(ppos - gt[None], axis=-1).mean(axis=1)  # (Kp,)
            m = motion(ppos)
            row = dict(model=label, train_stage=stage, sample=i, err_min=round(float(e.min()), 6),
                       err_mean=round(float(e.mean()), 6), pred_adj_disp_median=round(m["adj_disp_median"], 6),
                       pred_adj_disp_p95=round(m["adj_disp_p95"], 6), pred_net_disp_median=round(m["net_disp_median"], 6))
            writers["pred"].writerow(row)
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
    return summ


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda:0"
    samples = build_samples(os.path.join(args.out_dir, "x2_samples.pt"))
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
                                                           "pred_net_disp_median"]),
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
            e = s["pos_err_mean_of_mean"]
            print(f"[X2-SMOKE-3] {label} mean 32-step position error = {e*100:.2f} cm (<= 3 cm) -> "
                  f"{'PASS' if e <= 0.03 else 'FAIL (data reading/conversion differs from VLA training data)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="label=checkpoint path")
    p.add_argument("--out_dir", default="results/R1/R1X")
    main(p.parse_args())
