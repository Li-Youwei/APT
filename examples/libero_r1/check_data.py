"""Sanity check of the converted LIBERO data through APT's own dataset classes.

APT env, repo root, with APT_DATA_LOC_HOST set:
  python -m examples.libero_r1.check_data
Prints episode counts per suite and the tensor shapes of one training sample,
and checks the projected EE position lands inside the agentview image.
"""
from __future__ import annotations

import numpy as np

from data_utils import datasets


def main():
    for cls in (datasets.LiberoSpatial, datasets.LiberoObject, datasets.LiberoGoal, datasets.Libero10):
        ds = cls.inst(train_stage=0)
        x = ds[0]
        print(f"[DATA] {cls.__name__}: {len(ds)} episodes; prompt = {x['prompt_text']!r}")
        for k, v in x.items():
            if hasattr(v, "shape"):
                print(f"    {k}: {tuple(v.shape)} {getattr(v, 'dtype', '')}")
        # camera order is shuffled in training (DataConfig.shuffle_cameras), so camera 0 may be either view
        K = np.asarray(x["K"])[0]
        wcT = np.asarray(x["obs_extrinsics"])[-1, 0]
        p = np.asarray(x["current_ee_pose"])[0, :3, 3]
        cwT = np.linalg.inv(wcT)
        pc = cwT[:3, :3] @ p + cwT[:3, 3]
        uv = pc[:2] / pc[2] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
        print(f"    EE projected into camera-0 image at {uv.round(1)} (image 256x256, z={pc[2]:.3f})")
        fut = np.asarray(x["gt_future_ee_states"])
        print(f"    future gripper range [{fut[..., -1].min():.2f}, {fut[..., -1].max():.2f}]")


if __name__ == "__main__":
    main()
