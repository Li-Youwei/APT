"""Trajectory distance, clustering, candidate generation and selectors (R1 spec 2.3-2.4).

Pure numpy. Every number here is fixed by the spec; do not tune.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# ── spec 2.3 / 2.4 constants ────────────────────────────────────────────────
GRIPPER_OPEN_THRESH = 0.8      # openness > 0.8 -> open
GRIPPER_MISMATCH_STEPS = 8     # > 8 differing steps -> distance 1.0 m
GRIPPER_MISMATCH_DIST = 1.0
CLUSTER_THRESH = 0.08          # complete-linkage merge threshold (m)
COVER_THRESH = 0.08            # R1-D coverage threshold (m)
MAX_CANDIDATES = 6
TRAJ_LEN = 32
K_SAMPLES = 32
REPLAN_EVERY = 8


def binarize_gripper(grip: np.ndarray) -> np.ndarray:
    """True = open."""
    return np.asarray(grip) > GRIPPER_OPEN_THRESH


def _dist_from_parts(max_pos: np.ndarray, n_mismatch: np.ndarray) -> np.ndarray:
    return np.where(n_mismatch > GRIPPER_MISMATCH_STEPS, GRIPPER_MISMATCH_DIST, max_pos)


def traj_distance(pos_a, grip_a, pos_b, grip_b) -> float:
    """Spec 2.4: max per-step Euclidean distance of EE positions; 1.0 m if the
    binarised grippers differ on more than 8 steps.

    pos_*: (T, 3) world-frame metres; grip_*: (T,) openness in [0, 1].
    """
    pos_a = np.asarray(pos_a, np.float64)
    pos_b = np.asarray(pos_b, np.float64)
    assert pos_a.shape == pos_b.shape and pos_a.ndim == 2 and pos_a.shape[1] == 3
    max_pos = np.linalg.norm(pos_a - pos_b, axis=-1).max()
    n_mis = np.count_nonzero(binarize_gripper(grip_a) != binarize_gripper(grip_b))
    return float(_dist_from_parts(np.asarray(max_pos), np.asarray(n_mis)))


def pairwise_distances(pos: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """pos (N, T, 3), grip (N, T) -> (N, N) symmetric, zero diagonal."""
    pos = np.asarray(pos, np.float64)
    gb = binarize_gripper(grip)
    diff = pos[:, None] - pos[None, :]                      # (N, N, T, 3)
    max_pos = np.linalg.norm(diff, axis=-1).max(axis=-1)    # (N, N)
    n_mis = (gb[:, None] != gb[None, :]).sum(axis=-1)       # (N, N)
    D = _dist_from_parts(max_pos, n_mis)
    np.fill_diagonal(D, 0.0)
    return D


def distances_to_ref(pos: np.ndarray, grip: np.ndarray, ref_pos: np.ndarray, ref_grip: np.ndarray) -> np.ndarray:
    """pos (N, T, 3), grip (N, T), ref_pos (T, 3), ref_grip (T,) -> (N,)."""
    pos = np.asarray(pos, np.float64)
    max_pos = np.linalg.norm(pos - np.asarray(ref_pos, np.float64)[None], axis=-1).max(axis=-1)
    n_mis = (binarize_gripper(grip) != binarize_gripper(ref_grip)[None]).sum(axis=-1)
    return _dist_from_parts(max_pos, n_mis)


def complete_linkage(D: np.ndarray, thresh: float = CLUSTER_THRESH) -> List[List[int]]:
    """Agglomerative complete-linkage clustering cut at `thresh`.

    Repeatedly merges the two clusters with the smallest complete-link
    distance (max pairwise member distance) while that distance <= thresh.
    Ties: the pair whose (smallest member, smallest member) is lexicographically
    smallest. Deterministic. Returns clusters as sorted member lists, ordered
    by their smallest member.
    """
    D = np.asarray(D, np.float64)
    n = D.shape[0]
    members: Dict[int, List[int]] = {i: [i] for i in range(n)}  # key = smallest member
    CD = D.copy()
    np.fill_diagonal(CD, np.inf)
    active = np.ones(n, dtype=bool)
    while active.sum() > 1:
        M = np.where(active[:, None] & active[None, :], CD, np.inf)
        M = np.triu(M, k=1) + np.tril(np.full_like(M, np.inf))   # i < j only
        flat = int(np.argmin(M))                                  # row-major first min
        i, j = divmod(flat, n)
        if not (M[i, j] <= thresh):
            break
        # merge j into i (i < j, so i stays the smallest member)
        members[i] = sorted(members[i] + members.pop(j))
        CD[i, :] = np.maximum(CD[i, :], CD[j, :])
        CD[:, i] = CD[i, :]
        CD[i, i] = np.inf
        active[j] = False
        CD[j, :] = np.inf
        CD[:, j] = np.inf
    return [members[k] for k in sorted(members)]


def cluster_center(D: np.ndarray, members: Sequence[int]) -> int:
    """Member with the smallest sum of distances to the other members; ties -> smallest index."""
    members = sorted(members)
    sub = D[np.ix_(members, members)]
    sums = sub.sum(axis=1)
    return members[int(np.argmin(sums))]  # argmin returns first occurrence -> smallest index


@dataclass
class ClusterInfo:
    members: List[int]
    center: int

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class CandidateSet:
    clusters: List[ClusterInfo]          # ALL clusters, sorted by size desc, then min member asc
    candidates: List[ClusterInfo]        # first MAX_CANDIDATES of `clusters`
    D: np.ndarray = field(repr=False)    # (N, N) pairwise distances

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def cluster_sizes(self) -> List[int]:
        return [c.size for c in self.clusters]

    def cluster_rank_of_sample(self, sample_idx: int) -> int:
        """0-based index (in the size-sorted cluster list) of the cluster holding sample_idx."""
        for r, c in enumerate(self.clusters):
            if sample_idx in c.members:
                return r
        raise KeyError(sample_idx)


def sort_clusters(D: np.ndarray, clusters: List[List[int]]) -> List[ClusterInfo]:
    infos = [ClusterInfo(members=sorted(m), center=cluster_center(D, m)) for m in clusters]
    infos.sort(key=lambda c: (-c.size, c.members[0]))
    return infos


def make_candidates(pos: np.ndarray, grip: np.ndarray, thresh: float = CLUSTER_THRESH,
                    max_candidates: int = MAX_CANDIDATES) -> CandidateSet:
    """Spec 2.3: cluster the K samples, take each cluster's centre sample, keep at
    most `max_candidates` by cluster size desc (ties: smaller min member first)."""
    D = pairwise_distances(pos, grip)
    clusters = sort_clusters(D, complete_linkage(D, thresh))
    return CandidateSet(clusters=clusters, candidates=clusters[:max_candidates], D=D)


def n_clusters_at(D: np.ndarray, thresh: float) -> int:
    return len(complete_linkage(D, thresh))


# ── selectors (spec 2.3). One interface so the phase-2 Jev selector drops in. ──

@dataclass
class SelectionContext:
    """Everything a selector may look at for one query."""
    cand: CandidateSet
    pos: np.ndarray            # (K, T, 3) all VA samples
    grip: np.ndarray           # (K, T)
    instruction: str
    ref_pos: Optional[np.ndarray] = None   # (T, 3) VLA reference, if the arm queried it
    ref_grip: Optional[np.ndarray] = None  # (T,)
    extras: dict = field(default_factory=dict)  # e.g. images for the phase-2 Jev selector


@dataclass
class Selection:
    sample_idx: int                     # which of the K samples is executed
    candidate_idx: Optional[int]        # index into the size-sorted cluster list (None = n/a)
    info: dict = field(default_factory=dict)  # extra fields merged into the query log


class Selector:
    name: str = ""
    needs_vla_ref: bool = False   # evaluator queries the VLA reference before select()
    uses_candidates: bool = True

    def select(self, ctx: SelectionContext) -> Selection:  # pragma: no cover
        raise NotImplementedError


class RandomSelector(Selector):
    """Arm A (`va_random`): execute sample 0 as is, no clustering involved."""
    name = "va_random"
    uses_candidates = False

    def select(self, ctx: SelectionContext) -> Selection:
        return Selection(sample_idx=0, candidate_idx=ctx.cand.cluster_rank_of_sample(0),
                         info={"selected_is_cluster_center": int(0 in [c.center for c in ctx.cand.clusters])})


class OracleSelector(Selector):
    """Arm C (`va_oracle`): candidate centre closest to the VLA reference trajectory.
    Ties -> smaller candidate index (larger cluster)."""
    name = "va_oracle"
    needs_vla_ref = True

    def select(self, ctx: SelectionContext) -> Selection:
        assert ctx.ref_pos is not None
        centers = [c.center for c in ctx.cand.candidates]
        d = distances_to_ref(ctx.pos[centers], ctx.grip[centers], ctx.ref_pos, ctx.ref_grip)
        k = int(np.argmin(d))
        return Selection(sample_idx=centers[k], candidate_idx=k,
                         info={"oracle_candidate": k,
                               "cand_dists_to_vla": [float(x) for x in d],
                               "oracle_dist": float(d[k])})


class JevSelector(Selector):
    """Arm B (`va_jev`) -- phase 2, not implemented in phase 1.

    Phase-2 contract (spec 2.5): with 1 candidate execute it without a call;
    otherwise call Jev 3x with permuted options, average aligned probabilities,
    take the argmax over opt_1..opt_N (ties -> larger cluster), fall back to the
    largest cluster when all calls fail. It also needs the VLA reference for
    logging only (`needs_vla_ref = True`), never for the choice.
    """
    name = "va_jev"
    needs_vla_ref = True

    def select(self, ctx: SelectionContext) -> Selection:
        raise NotImplementedError("va_jev is phase 2 (spec section 2.5); not implemented yet")


SELECTORS = {
    "va_random": RandomSelector,
    "va_oracle": OracleSelector,
    "va_jev": JevSelector,
}


def dist_selected_to_prev(prev_pos: Optional[np.ndarray], cur_pos: np.ndarray, shift: int = REPLAN_EVERY) -> Optional[float]:
    """Spec 4.3: max position distance between the first T-shift steps of the
    current selection and steps shift+1..T of the previous one (metres)."""
    if prev_pos is None:
        return None
    T = cur_pos.shape[0]
    return float(np.linalg.norm(np.asarray(cur_pos[: T - shift], np.float64)
                                - np.asarray(prev_pos[shift:], np.float64), axis=-1).max())


def wilson_interval(k: int, n: int, z: float = 1.959963984540054):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


# ── R1-D statistics (spec 2.2 / 4.2) ───────────────────────────────────────

def r1d_analyse(pos, grip, ref_pos, ref_grip):
    """All R1-D statistics for one (state, checkpoint). Pure numpy."""
    D = pairwise_distances(pos, grip)
    clusters = sort_clusters(D, complete_linkage(D, CLUSTER_THRESH))
    cand = CandidateSet(clusters=clusters, candidates=clusters[:MAX_CANDIDATES], D=D)
    d_ref = distances_to_ref(pos, grip, ref_pos, ref_grip)
    nearest = int(np.argmin(d_ref))
    rank0 = cand.cluster_rank_of_sample(nearest)
    n = pos.shape[0]
    return dict(
        n_clusters_004=n_clusters_at(D, 0.04),
        n_clusters_008=cand.n_clusters,
        n_clusters_012=n_clusters_at(D, 0.12),
        cluster_sizes_008=json.dumps(cand.cluster_sizes),
        covered=int(d_ref.min() <= COVER_THRESH),
        min_dist_to_vla=round(float(d_ref.min()), 6),
        vla_cluster_rank=rank0 + 1,
        vla_cluster_share=round(clusters[rank0].size / n, 6),
    ), cand


# ── R1-X motion statistics (spec 9.2) ───────────────────────────────────────

def motion_stats(pos: np.ndarray, grip: np.ndarray, cur_pos: Optional[np.ndarray] = None) -> dict:
    """Smoothness statistics of K trajectories, all in metres.

    pos (K, T, 3), grip (K, T), cur_pos (3,) = EE position at query time.
      adj_disp_median / adj_disp_p95 : over all K x (T-1) steps |p[t+1] - p[t]|
      adj_disp_max                   : max of the same
      net_disp_median                : median over K of |p[T-1] - cur_pos| ("32-step net
                                       displacement"; falls back to |p[T-1] - p[0]|
                                       when cur_pos is None)
      net_disp_first_last_median     : median over K of |p[T-1] - p[0]|
      grip_std_median                : median over K of std_t(grip)
    """
    pos = np.asarray(pos, np.float64)
    adj = np.linalg.norm(np.diff(pos, axis=1), axis=-1).ravel()
    fl = np.linalg.norm(pos[:, -1] - pos[:, 0], axis=-1)
    net = fl if cur_pos is None else np.linalg.norm(pos[:, -1] - np.asarray(cur_pos, np.float64)[None], axis=-1)
    return dict(
        adj_disp_median=float(np.median(adj)),
        adj_disp_p95=float(np.percentile(adj, 95)),
        adj_disp_max=float(adj.max()),
        net_disp_median=float(np.median(net)),
        net_disp_first_last_median=float(np.median(fl)),
        grip_std_median=float(np.median(np.std(np.asarray(grip, np.float64), axis=1))),
    )
