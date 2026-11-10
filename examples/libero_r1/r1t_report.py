"""R1-T outputs (spec 4.1) from the trainer's TensorBoard logs and resource logs.

APT env (needs `tensorboard`), repo root:
  python -m examples.libero_r1.r1t_report --log_dir logs/APT/va_libero4_r1 \
      --ckpt_dir checkpoints/APT/va_libero4_r1 --out_dir results/R1/R1T [--every 1000]

Writes
  loss.csv          step, pos_loss, rot_loss, openness_loss, total_loss, lr at every
                    `--every` steps (each value is the trainer's mean over the
                    preceding `log_interval` iterations, averaged over ranks)
  timing_memory.csv iter, rank, wall_clock_s, peak allocated / reserved GB
  config.json       copy of the run's TrainConfig dump
  tensorboard/      copy of the event files
For the smoke run use `--every 1` (trained with --log_interval 1) and
`--smoke_check` (spec section 7, ruling 7): PASS iff the mean total_loss of
steps 181-200 is lower than the mean of steps 1-20.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys

KEYS = ["pos_loss", "rot_loss", "openness_loss", "total_loss"]


def load_scalars(log_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        out[tag] = {e.step: e.value for e in acc.Scalars(tag)}
    return out


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    sc = load_scalars(args.log_dir)
    steps = sorted(sc.get("train/total_loss", {}))
    if not steps:
        raise SystemExit(f"no train/total_loss scalars in {args.log_dir}")
    rows = []
    for s in steps:
        if s % args.every:
            continue
        row = {"step": s}
        for k in KEYS:
            row[k] = sc.get("train/" + k, {}).get(s)
        row["lr"] = sc.get("lr", {}).get(s)
        rows.append(row)
    with open(os.path.join(args.out_dir, "loss.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["step"] + KEYS + ["lr"])
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] loss.csv: {len(rows)} rows, steps {rows[0]['step']}..{rows[-1]['step']}")

    res = []
    for f in sorted(glob.glob(os.path.join(args.log_dir, "resource_rank*.jsonl"))):
        with open(f) as fp:
            res += [json.loads(l) for l in fp if l.strip()]
    if res:
        res.sort(key=lambda r: (r["iter"], r["rank"]))
        with open(os.path.join(args.out_dir, "timing_memory.csv"), "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(res[0].keys()))
            w.writeheader()
            w.writerows(res)
        print(f"[OK] timing_memory.csv: {len(res)} rows")
    else:
        print("[WARN] no resource_rank*.jsonl found")

    if args.ckpt_dir:
        for j in glob.glob(os.path.join(args.ckpt_dir, "*.json")):
            shutil.copy(j, os.path.join(args.out_dir, "config_" + os.path.basename(j)))
    tb_out = os.path.join(args.out_dir, "tensorboard")
    os.makedirs(tb_out, exist_ok=True)
    for f in glob.glob(os.path.join(args.log_dir, "events.out.tfevents.*")):
        shutil.copy(f, tb_out)

    if args.smoke_check:
        tl = sc["train/total_loss"]
        early = [tl[s] for s in range(1, 21) if s in tl]
        late = [tl[s] for s in range(181, 201) if s in tl]
        if len(early) != 20 or len(late) != 20:
            print(f"[SMOKE-2] FAIL: need per-step total_loss for steps 1-20 and 181-200 "
                  f"(found {len(early)} and {len(late)}); train with --log_interval 1")
            sys.exit(1)
        a, b = sum(early) / 20, sum(late) / 20
        ok = b < a
        print(f"[SMOKE-2] mean total_loss steps 1-20 = {a:.4f}, steps 181-200 = {b:.4f} -> {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--ckpt_dir", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--every", type=int, default=1000)
    p.add_argument("--smoke_check", action="store_true")
    main(p.parse_args())
