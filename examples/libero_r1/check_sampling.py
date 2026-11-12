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
and prints the timing of both paths. Exit code 1 if 1 or 2 fail. [3] is
information only (spec section 8, ruling 4): its WARN above 5 mm is expected
bf16 amplification, not a failure; formal runs always use the shared path.
[4] (diagnostic, not a pass criterion) shows how far the same noise lands
under the two reference cameras -- the mechanism behind the 2026-11-11 failure.

Smoke-11 diagnostic (not a pass criterion; run on the VA server, then on the
VLA server as a positive control):
  python -m examples.libero_r1.check_sampling --uri va  --diag11
  python -m examples.libero_r1.check_sampling --uri vla --diag11
Compares the true instruction with (a) the spec's "do nothing", (b) "do
nothing" padded to the SAME VLM token count, (c) other libero_goal
instructions with the same token count. Equal token count means identical
tensor shapes, so a VA without any language path must give a bit-exact (0)
difference; a length mismatch alone changes kernel shapes and bf16 rounding,
which the multimodal VA amplifies (compare check [3]).

Smoke test 11 (spec section 8, rulings 1-3), run on BOTH servers:
  python -m examples.libero_r1.check_sampling --uri va  --smoke11   # VA 40k
  python -m examples.libero_r1.check_sampling --uri vla --smoke11   # positive control
Control prompt = "do nothing" padded with " now" to the SAME VLM token count as
the true instruction; same observation, noise and reference camera (both
cameras checked). PASS on the VA server (train_stage 0): max element-wise
position difference <= 1e-4 m for every camera. PASS on the VLA server
(train_stage 1): difference > 0.01 m for every camera. FAIL on the VA side
means language leaks into the VA: stop and report.
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
    if args.diag11:
        diag11(pol, frame, instr, noise, suite)
        sys.exit(0)
    if args.smoke11:
        sys.exit(0 if smoke11(pol, frame, instr, noise) else 1)

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
        tag = "OK" if np.median(d_share) <= 0.005 else "WARN (info only, spec 8.4)"
        print(f"[3] ref_cam={cam} shared VLM vs replicated obs: per-sample max pos diff median "
              f"{np.median(d_share)*1000:.2f} mm, max {d_share.max()*1000:.2f} mm; gripper max diff "
              f"{np.abs(a['grip']-c['grip']).max():.3f} -> {tag}")
        print(f"[T] ref_cam={cam} server infer time, K={args.K}: shared {a['infer_time_s']:.3f}s / "
              f"{b['infer_time_s']:.3f}s, replicated {c['infer_time_s']:.3f}s")

    d_cam = np.linalg.norm(first[REF_CAMS[0]]["pos"] - first[REF_CAMS[1]]["pos"], axis=-1).max(axis=-1)
    print(f"[4] diagnostic: same noise, agentview vs eye_in_hand reference: per-sample max pos diff median "
          f"{np.median(d_cam)*1000:.2f} mm, max {d_cam.max()*1000:.2f} mm (expected large; explains the old [1] FAIL)")
    sys.exit(0 if ok else 1)


SMOKE11_VA_MAX = 1e-4    # m, spec 8.2
SMOKE11_VLA_MIN = 0.01   # m, spec 8.2 positive control
PAD_WORD = " now"


def length_matched_control(pol, frame, instr, cam, base="do nothing", max_pad=60):
    """`base` + PAD_WORD * k with the same VLM token count as `instr`, or None."""
    one = np.zeros((1, pol.meta["nee"], pol.meta["ta"], pol.meta["act_dim"]), np.float32)
    target = pol.sample(frame, instr, one, cam)["seq_len"]
    assert target is not None, "server does not report seq_len (old server code still running?)"
    for k in range(max_pad + 1):
        cand = base + PAD_WORD * k
        n = pol.sample(frame, cand, one, cam)["seq_len"]
        if n == target:
            return cand, target
        if n > target:
            break
    return None, target


def smoke11(pol, frame, instr, noise) -> bool:
    stage = pol.meta["train_stage"]
    assert stage in (0, 1), pol.meta
    print(f"[INFO] smoke 11 on train_stage={stage} server, ckpt={pol.meta.get('ckpt_path')} "
          f"(iter {pol.meta.get('current_iters')})")
    ok = True
    for cam in REF_CAMS:
        ctrl, n_tok = length_matched_control(pol, frame, instr, cam)
        if ctrl is None:
            print(f"[SMOKE-11] ref_cam={cam}: cannot pad 'do nothing' to exactly {n_tok} tokens -> FAIL (cannot evaluate)")
            ok = False
            continue
        a = pol.sample(frame, instr, noise, cam)
        z = pol.sample(frame, ctrl, noise, cam)
        assert a["seq_len"] == z["seq_len"] == n_tok, (a["seq_len"], z["seq_len"], n_tok)
        d = float(np.abs(a["pos"] - z["pos"]).max())
        dg = float(np.abs(a["grip"] - z["grip"]).max())
        if stage == 0:
            passed, rule = d <= SMOKE11_VA_MAX, f"VA: <= {SMOKE11_VA_MAX:g} m"
        else:
            passed, rule = d > SMOKE11_VLA_MIN, f"VLA positive control: > {SMOKE11_VLA_MIN:g} m"
        ok &= bool(passed)
        print(f"[SMOKE-11] ref_cam={cam}: {instr!r} vs {ctrl!r} ({n_tok} tokens each): max |diff| of positions = "
              f"{d:.3e} m ({d*1000:.3f} mm), gripper {dg:.3e}; rule {rule} -> {'PASS' if passed else 'FAIL'}")
    print(f"[SMOKE-11] overall ({'VA' if stage == 0 else 'VLA'} side): {'PASS' if ok else 'FAIL'}")
    return ok


def diag11(pol, frame, instr, noise, suite):
    cam = REF_CAMS[0]
    one = noise[:1]

    def seq_len(prompt):
        return pol.sample(frame, prompt, one, cam)["seq_len"]

    def compare(tag, prompt, base):
        o = pol.sample(frame, prompt, noise, cam)
        d = np.abs(base["pos"] - o["pos"]).max()
        dg = np.abs(base["grip"] - o["grip"]).max()
        print(f"[D11] {tag:<34} tokens {base['seq_len']} vs {o['seq_len']}: max |pos diff| = {d:.3e} m "
              f"({d*1000:.3f} mm), gripper {dg:.3e}{'  BIT-EXACT' if d == 0 and dg == 0 else ''}  prompt={prompt!r}")

    print(f"[INFO] server train_stage={pol.meta['train_stage']} ckpt={pol.meta.get('ckpt_path')} ref_cam={cam}")
    base = pol.sample(frame, instr, noise, cam)
    assert base["seq_len"] is not None, "server does not report seq_len (old server code still running?)"
    base2 = pol.sample(frame, instr, noise, cam)
    print(f"[D11] repeat of the true instruction: max |pos diff| = {np.abs(base['pos']-base2['pos']).max():.3e} m")
    target = base["seq_len"]
    compare("(a) spec control", "do nothing", base)
    matched = None
    for k in range(0, 40):
        cand = "do nothing" + " now" * k
        n = seq_len(cand)
        if n == target:
            matched = cand
            break
        if n > target:
            break
    if matched is None:
        print(f"[D11] (b) no 'do nothing now...' padding reaches exactly {target} tokens")
    else:
        compare("(b) 'do nothing' length-matched", matched, base)
    n_other = 0
    for t in range(suite.n_tasks):
        other = suite.get_task(t).language
        if other == instr:
            continue
        if seq_len(other) == target:
            compare(f"(c) other goal task {t}, same length", other, base)
            n_other += 1
    if n_other == 0:
        print(f"[D11] (c) no other libero_goal instruction has exactly {target} tokens")
    print("[D11] reading: VA (train_stage 0) without a language path -> (b)/(c) BIT-EXACT; "
          "VLA (train_stage 1) -> (b)/(c) clearly non-zero (positive control).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="va")
    p.add_argument("--ns_host", default="localhost")
    p.add_argument("--ns_port", type=int, default=9091)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--smoke11", action="store_true",
                   help="run only smoke test 11 (token-matched language check; VA and VLA servers)")
    p.add_argument("--diag11", action="store_true", help="length-matched language-invariance diagnostic")
    main(p.parse_args())
