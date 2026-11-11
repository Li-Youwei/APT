"""Thin client for the APT inference service's stateless R1 API.

Runs in the LIBERO env: only numpy + shm_transport. Noise is generated here,
from the per-episode seed, so results do not depend on server-side RNG state
and several clients can share one server.
"""
from __future__ import annotations

import time

import numpy as np

from shm_transport import get_shm_proxy

DATA_CONFIG = "Libero"
# Candidate action-reference cameras of the Libero data config
# (data_utils/datasets.py: ee_ref_cams={0: ("agentview", "eye_in_hand")}).
# Training draws one uniformly per sample; we keep that distribution but draw
# it from the episode's seeded stream instead of the server's unseeded `random`.
REF_CAMS = ("agentview", "eye_in_hand")


def noise_streams(seed: int):
    """Two independent numpy generators derived from one integer seed:
    (VA stream, VLA stream). Same seed -> same noise, whatever the arm."""
    va_ss, vla_ss = np.random.SeedSequence(int(seed)).spawn(2)
    return np.random.default_rng(va_ss), np.random.default_rng(vla_ss)


def ref_cam_streams(seed: int):
    """Two more independent generators from the same seed, for the per-query
    reference-camera draw: (VA, VLA). SeedSequence children are keyed by index,
    so children 0-1 (the noise streams above) are unchanged."""
    _a, _b, va_ss, vla_ss = np.random.SeedSequence(int(seed)).spawn(4)
    return np.random.default_rng(va_ss), np.random.default_rng(vla_ss)


def draw_ref_cam(rng: np.random.Generator) -> str:
    return REF_CAMS[int(rng.integers(len(REF_CAMS)))]


class Policy:
    def __init__(self, uri: str, ns_port: int, ns_host: str = "localhost"):
        self.uri = uri
        self.proxy = get_shm_proxy(uri, ns_host=ns_host, ns_port=ns_port)
        self.meta = self.proxy.get_meta()

    def refresh_meta(self):
        self.meta = self.proxy.get_meta()
        return self.meta

    def load_weights(self, ckpt_path: str):
        self.meta = self.proxy.load_weights(ckpt_path)
        return self.meta

    def noise(self, rng: np.random.Generator, K: int) -> np.ndarray:
        shape = (K, self.meta["nee"], self.meta["ta"], self.meta["act_dim"])
        return rng.standard_normal(shape).astype(np.float32)

    def sample(self, frame: dict, prompt: str, noise: np.ndarray, ref_cam: str, share_vlm: bool = True) -> dict:
        assert ref_cam in REF_CAMS, ref_cam
        t0 = time.perf_counter()
        out = self.proxy.sample_trajs(frame, prompt, noise, DATA_CONFIG, bool(share_vlm), ref_cam)
        assert out.get("ref_cam") == ref_cam, "server ignored ref_cam (old server code still running?)"
        rpc_time = time.perf_counter() - t0
        poses = np.asarray(out["ee_pose"], np.float64)  # (K, T, 4, 4)
        return {
            "poses": poses,
            "pos": poses[..., :3, 3],
            "grip": np.asarray(out["gripper"], np.float64),  # (K, T)
            "infer_time_s": float(out["infer_time_s"]),
            "rpc_time_s": rpc_time,
            "ref_cam": ref_cam,
        }
