"""Projection and drawing helpers for R1 figures and videos (numpy + cv2)."""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional, Sequence

import cv2
import numpy as np

# distinct BGR colours for clusters (rank 0 = largest cluster)
PALETTE_BGR = [
    (75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0), (48, 130, 245),
    (180, 30, 145), (240, 240, 70), (230, 50, 240), (60, 245, 210), (212, 190, 250),
    (128, 128, 0), (255, 190, 220), (40, 110, 170), (200, 250, 255), (0, 0, 128),
    (195, 255, 170), (0, 128, 128), (180, 215, 255), (128, 0, 0), (128, 128, 128),
]


def project(K: np.ndarray, wcT: np.ndarray, pts: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """World points (..., 3) -> pixel coords (..., 2) on the (H-flipped) APT image,
    same convention as examples/libero/eval_libero_pred.py::proj. `scale`
    multiplies pixel coords (2 for a 2x upscaled image)."""
    cwT = np.linalg.inv(wcT)
    pc = np.asarray(pts, np.float64) @ cwT[:3, :3].T + cwT[:3, 3]
    z = np.where(np.abs(pc[..., 2:3]) < 1e-9, 1e-9, pc[..., 2:3])
    xy = pc[..., :2] / z
    uv = xy * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv * scale


def clip_to_image(uv: np.ndarray, W: int, H: int) -> np.ndarray:
    out = uv.copy()
    out[..., 0] = np.clip(out[..., 0], 0, W - 1)
    out[..., 1] = np.clip(out[..., 1], 0, H - 1)
    return out


def upscale2x_rgb(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, (rgb.shape[1] * 2, rgb.shape[0] * 2), interpolation=cv2.INTER_CUBIC)


def _poly(img, uv, color, thickness):
    pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, color, thickness, lineType=cv2.LINE_AA)


def draw_r1d_figure(rgb256: np.ndarray, K: np.ndarray, wcT: np.ndarray,
                    samples_pos: np.ndarray, sample_cluster_rank: Sequence[int],
                    ref_pos: Optional[np.ndarray], title: str = "") -> np.ndarray:
    """Spec 4.2 figure: 512x512 agentview, every sample drawn in its cluster's
    colour (by size rank), VLA reference as a thick white line. Returns BGR."""
    img = cv2.cvtColor(upscale2x_rgb(rgb256), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    order = np.argsort(-np.asarray(sample_cluster_rank))  # draw small clusters first, big on top
    for i in order:
        uv = clip_to_image(project(K, wcT, samples_pos[i], scale=2.0), W, H)
        _poly(img, uv, PALETTE_BGR[int(sample_cluster_rank[i]) % len(PALETTE_BGR)], 1)
    if ref_pos is not None:
        uv = clip_to_image(project(K, wcT, ref_pos, scale=2.0), W, H)
        _poly(img, uv, (0, 0, 0), 6)
        _poly(img, uv, (255, 255, 255), 4)
    # start point (current EE) for the projection sanity check (smoke test 4)
    start = clip_to_image(project(K, wcT, samples_pos[0, 0], scale=2.0), W, H)
    cv2.circle(img, tuple(np.round(start).astype(int).tolist()), 5, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(start).astype(int).tolist()), 3, (255, 255, 255), -1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def video_frame(obs: dict, Ks, wcTs, traj_pos: Optional[np.ndarray], color=(255, 0, 255),
                points_only: bool = False, text: str = "") -> np.ndarray:
    """agentview | eye_in_hand side by side (BGR), optionally with a trajectory overlay."""
    tiles = []
    for cam, K, wcT in zip(("agentview_image", "robot0_eye_in_hand_image"), Ks, wcTs):
        bgr = np.ascontiguousarray(obs[cam][::-1, :, ::-1])
        if traj_pos is not None and len(traj_pos):
            uv = project(K, wcT, traj_pos)
            H, W = bgr.shape[:2]
            if points_only:
                for p in uv:
                    if 0 <= p[0] < W and 0 <= p[1] < H:
                        cv2.circle(bgr, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
            else:
                _poly(bgr, clip_to_image(uv, W, H), color, 2)
        tiles.append(bgr)
    frame = np.concatenate(tiles, axis=1)
    if text:
        cv2.putText(frame, text, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class VideoWriter:
    """mp4v via OpenCV, re-encoded to H.264 with ffmpeg when available."""

    def __init__(self, path: str, fps: float = 20.0):
        self.path = path
        self.fps = fps
        self.tmp = path + ".tmp.mp4"
        self.writer = None

    def write(self, bgr: np.ndarray):
        if self.writer is None:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            H, W = bgr.shape[:2]
            self.writer = cv2.VideoWriter(self.tmp, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (W, H))
        self.writer.write(bgr)

    def close(self):
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None
        ok = False
        if shutil.which("ffmpeg"):
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", self.tmp, "-vcodec", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "23", self.path],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok = r.returncode == 0
        if ok:
            os.remove(self.tmp)
        else:
            os.replace(self.tmp, self.path)
