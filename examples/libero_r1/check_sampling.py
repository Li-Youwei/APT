"""Engineering check of the K-sample API (spec 3.3). LIBERO env, repo root,
with a VA server (and optionally a VLA server) running:

  python -m examples.libero_r1.check_sampling --uri va --ns_port 9091

On libero_goal task 0 state 0, for EACH action-reference camera
(agentview, eye_in_hand; passed explicitly, see TrajPlanner.sample_trajs), it
checks that
  1. the same noise gives identical samples on repeated calls (reproducibility),
  2. different noise rows give different samples (independent noise),
  3. the shared-VLM path (one VLM pass, outputs repeated K times) matches the
     replicated-observation path (old `sample_num` behaviour) up to bf16 noise,
and prints the timing of both paths. Exit code 1 if 1 or 2 fail; 3 is reported
(WARN above 5 mm) because bf16 kernels may differ with batch layout.
[4] (diagnostic, not a pass criterion) shows how far the same noise lands
under the two reference cameras -- the mechanism behind the 2026-11-11 failure.

Smoke test 11 (spec section 7, ruling 2), VA server only:
  python -m examples.libero_r1.check_sampling --uri va --smoke11
same observation, same noise, prompt = true instruction vs "do nothing":
PASS iff the max element-wise difference of the 32 returned EE positions is
<= 1e-4 m. FAIL means language leaks into the VA: stop and report.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from shm_transport import setup_log_level

from . import libero_common as LC
from .policy_client import REF_CAMS, Policy, noise_streams


def main(args):
    setup_log_level("WARNING")
    pol = Policy(args.uri, args.ns_port, args.ns_host)
    print(f"[INFO] meta: {pol.meta}")
    suite = LC.get_suite("libero_goal")
    task = suite.get_task(0)
    env, instr = LC.make_env(task)
    obs = LC.reset_to_state(env, suite.get_task_init_states(0)[0])
    frame = LC.obs_libero2apt(obs, time=0, env=env)
    env.close()

    rng, _ = noise_streams(12345)
    noise = pol.noise(rng, args.K)
    if args.smoke11:
        assert pol.meta["train_stage"] == 0, f"smoke 11 needs the VA server: {pol.meta}"
    if args.smoke11:
        assert pol.meta["train_stage"] == 0, f"smoke 11 needs the VA server: {pol.meta}"
        ok = True
        for cam in REF_CAMS:
            a = pol.sample(frame, instr, noise, cam)
            z = pol.sample(frame, "do nothing", noise, cam)
            d = np.abs(a["pos"] - z["pos"]).max()
            dg = np.abs(a["grip"] - z["grip"]).max()
            ok &= bool(d <= 1e-4)
            print(f"[SMOKE-11] ref_cam={cam}: instruction {instr!r} vs 'do nothing': max |diff| of positions = "
                  f"{d:.3e} m (gripper max |diff| {dg:.3e}) -> {'PASS' if d <= 1e-4 else 'FAIL (language leak?)'}")
        sys.exit(0 if ok else 1)

    ok = True
    first = {}
    for cam in REF_CAMS:
        a = pol.sample(frame, instr, noise, cam, share_vlm=True)
        b = pol.sample(frame, instr, noise, cam, share_vlm=True)
        c = pol.sample(frame, instr, noise, cam, share_vlm=False)
        first[cam] = a

        d_rep = np.abs(a["poses"] - b["poses"]).max()
        d_rep_pos = np.linalg.norm(a["pos"] - b["pos"], axis=-1).max()
        print(f"[1] ref_cam={cam} repeat, same noise: max |diff| of 4x4 pose elements = {d_rep:.3e} "
              f"(positions {d_rep_pos:.3e} m) -> {'PASS' if d_rep <= 1e-5 else 'FAIL'}"
              f"{' (bit-exact)' if d_rep == 0 else ''}")
        ok &= bool(d_rep <= 1e-5)

        spread = np.linalg.norm(a["pos"] - a["pos"][:1], axis=-1).max(axis=-1)[1:]
        print(f"[2] ref_cam={cam} samples differ from sample 0: min/median max-pos-dist = {spread.min():.4f} / "
              f"{np.median(spread):.4f} m -> {'PASS' if spread.min() > 0 else 'FAIL'}")
        ok &= bool(spread.min() > 0)

        d_share = np.linalg.norm(a["pos"] - c["pos"], axis=-1).max(axis=-1)  # per sample, metres
        tag = "OK" if np.median(d_share) <= 0.005 else "WARN"
        print(f"[3] ref_cam={cam} shared VLM vs replicated obs: per-sample max pos diff median "
              f"{np.median(d_share)*1000:.2f} mm, max {d_share.max()*1000:.2f} mm; gripper max diff "
              f"{np.abs(a['grip']-c['grip']).max():.3f} -> {tag}")
        print(f"[T] ref_cam={cam} server infer time, K={args.K}: shared {a['infer_time_s']:.3f}s / "
              f"{b['infer_time_s']:.3f}s, replicated {c['infer_time_s']:.3f}s")

    d_cam = np.linalg.norm(first[REF_CAMS[0]]["pos"] - first[REF_CAMS[1]]["pos"], axis=-1).max(axis=-1)
    print(f"[4] diagnostic: same noise, agentview vs eye_in_hand reference: per-sample max pos diff median "
          f"{np.median(d_cam)*1000:.2f} mm, max {d_cam.max()*1000:.2f} mm (expected large; explains the old [1] FAIL)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="va")
    p.add_argument("--ns_host", default="localhost")
    p.add_argument("--ns_port", type=int, default=9091)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--smoke11", action="store_true", help="run only smoke test 11 (language-leak check)")
    main(p.parse_args())
