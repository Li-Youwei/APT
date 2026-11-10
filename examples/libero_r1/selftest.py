"""Offline unit checks for the pure-numpy parts (no GPU, no LIBERO, no torch):

  python -m examples.libero_r1.selftest

Covers traj_select (distance, gripper rule, complete linkage vs scipy,
centres, candidate ordering, selectors, prev-distance, Wilson) and the R1-D
statistics. scipy is optional (cross-check skipped if missing).
"""
from __future__ import annotations

import numpy as np

from . import traj_select as TS


def _traj(start, vel, T=32, grip_open=True, close_at=None):
    t = np.arange(1, T + 1)[:, None]
    pos = np.asarray(start)[None] + t * np.asarray(vel)[None]
    g = np.ones(T) if grip_open else np.zeros(T)
    if close_at is not None:
        g[close_at:] = 0.0
    return pos, g


def test_distance():
    p1, g1 = _traj([0, 0, 0], [0.001, 0, 0])
    p2, g2 = _traj([0, 0.05, 0], [0.001, 0, 0])
    assert abs(TS.traj_distance(p1, g1, p2, g2) - 0.05) < 1e-12
    # gripper: 8 differing steps -> still position distance; 9 -> 1.0
    g3 = g1.copy(); g3[-8:] = 0.0
    assert abs(TS.traj_distance(p1, g1, p2, g3) - 0.05) < 1e-12
    g3[-9:] = 0.0
    assert TS.traj_distance(p1, g1, p2, g3) == 1.0
    # threshold 0.8 is strict ">"
    assert TS.binarize_gripper(np.array([0.8, 0.8001])).tolist() == [False, True]
    # pairwise == scalar
    P = np.stack([p1, p2]); G = np.stack([g1, g3])
    D = TS.pairwise_distances(P, G)
    assert D[0, 1] == D[1, 0] == 1.0 and D[0, 0] == 0
    d = TS.distances_to_ref(P, G, p1, g1)
    assert d[0] == 0 and d[1] == 1.0


def test_linkage_simple():
    # three groups along y, spacing within a group 0.02 m, groups 0.3 m apart
    trajs = []
    for gy in (0.0, 0.3, 0.6):
        for k in range(3):
            trajs.append(_traj([0, gy + 0.02 * k, 0], [0.001, 0, 0])[0])
    P = np.stack(trajs); G = np.ones(P.shape[:2])
    D = TS.pairwise_distances(P, G)
    cl = TS.complete_linkage(D, 0.08)
    assert cl == [[0, 1, 2], [3, 4, 5], [6, 7, 8]], cl
    # chaining: 0, .05, .10 -> single linkage would merge all, complete linkage must not at 0.08
    P2 = np.stack([_traj([0, y, 0], [0, 0, 0])[0] for y in (0.0, 0.05, 0.10)])
    cl2 = TS.complete_linkage(TS.pairwise_distances(P2, np.ones(P2.shape[:2])), 0.08)
    assert len(cl2) == 2, cl2
    # threshold is inclusive (<=)
    P3 = np.stack([_traj([0, y, 0], [0, 0, 0])[0] for y in (0.0, 0.08)])
    assert len(TS.complete_linkage(TS.pairwise_distances(P3, np.ones(P3.shape[:2])), 0.08)) == 1


def test_linkage_vs_scipy(n_trials=200):
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        print("  (scipy missing: cross-check skipped)")
        return
    rng = np.random.default_rng(0)
    mismatches = 0
    for trial in range(n_trials):
        n = int(rng.integers(2, 65))
        n_modes = int(rng.integers(1, 6))
        centers = rng.uniform(-0.15, 0.15, size=(n_modes, 3))
        vel = rng.normal(0, 0.003, size=(n_modes, 3))
        lab = rng.integers(0, n_modes, size=n)
        T = 32
        t = np.arange(1, T + 1)[None, :, None]
        P = centers[lab][:, None] + t * vel[lab][:, None] + rng.normal(0, 0.012, size=(n, T, 3))
        G = (rng.random((n, T)) > 0.15).astype(float) if trial % 3 == 0 else np.ones((n, T))
        D = TS.pairwise_distances(P, G)
        for thr in (0.04, 0.08, 0.12):
            ours = sorted(sorted(c) for c in TS.complete_linkage(D, thr))
            Z = linkage(squareform(D, checks=False), method="complete")
            f = fcluster(Z, t=thr, criterion="distance")
            ref = sorted(sorted(np.flatnonzero(f == k).tolist()) for k in np.unique(f))
            if ours != ref:
                mismatches += 1
    assert mismatches == 0, f"{mismatches} partitions differ from scipy"
    print(f"  complete linkage == scipy fcluster on {n_trials} x 3 random cases")


def test_candidates_and_selectors():
    rng = np.random.default_rng(1)
    # 32 samples: mode sizes 12, 8, 8, 2, 1, 1 (ties 8/8 broken by min index)
    sizes = [2, 8, 12, 8, 1, 1]
    trajs, grips = [], []
    for m, s in enumerate(sizes):
        for _ in range(s):
            p, g = _traj([0, 0.3 * m, 0], [0.002, 0, 0])
            trajs.append(p + rng.normal(0, 0.002, size=p.shape)); grips.append(g)
    order = rng.permutation(len(trajs))
    P = np.stack(trajs)[order]; G = np.stack(grips)[order]
    cs = TS.make_candidates(P, G)
    assert cs.cluster_sizes == [12, 8, 8, 2, 1, 1], cs.cluster_sizes
    assert cs.clusters[1].members[0] < cs.clusters[2].members[0]
    assert len(cs.candidates) == 6
    # centre minimises the sum of distances, ties -> smallest index
    for c in cs.clusters:
        sub = cs.D[np.ix_(c.members, c.members)].sum(1)
        assert c.center == c.members[int(np.argmin(sub))]
    Dt = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    assert TS.cluster_center(Dt, [2, 0, 1]) == 0
    # max-candidate truncation
    many = TS.make_candidates(np.stack([_traj([0, 0.3 * k, 0], [0, 0, 0])[0] for k in range(9)]), np.ones((9, 32)))
    assert many.n_clusters == 9 and len(many.candidates) == 6
    # selectors
    ctx = TS.SelectionContext(cand=cs, pos=P, grip=G, instruction="x")
    a = TS.RandomSelector().select(ctx)
    assert a.sample_idx == 0 and a.candidate_idx == cs.cluster_rank_of_sample(0)
    target = cs.candidates[3]
    ctx.ref_pos, ctx.ref_grip = P[target.center] + 0.001, G[target.center]
    c = TS.OracleSelector().select(ctx)
    assert c.candidate_idx == 3 and c.sample_idx == target.center
    assert abs(c.info["oracle_dist"] - min(c.info["cand_dists_to_vla"])) < 1e-12
    try:
        TS.JevSelector().select(ctx)
        raise AssertionError("JevSelector should not be implemented in phase 1")
    except NotImplementedError:
        pass


def test_misc():
    p, _ = _traj([0, 0, 0], [0.01, 0, 0])
    assert TS.dist_selected_to_prev(None, p) is None
    # if the new plan continues the old one exactly, distance is 0
    prev = p
    cur = np.concatenate([p[8:], p[-1:] + np.zeros((8, 3))])
    assert TS.dist_selected_to_prev(prev, cur) == 0.0
    lo, hi = TS.wilson_interval(45, 50)
    assert abs(lo - 0.7864) < 1e-3 and abs(hi - 0.9565) < 1e-3, (lo, hi)
    assert TS.wilson_interval(0, 10)[0] == 0.0


def test_r1d_analyse():
    analyse = TS.r1d_analyse
    trajs = [_traj([0, 0.3 * (k % 2), 0], [0.001, 0, 0])[0] for k in range(64)]
    P = np.stack(trajs); G = np.ones((64, 32))
    st, cand = analyse(P, G, P[1] + 0.01, G[1])
    assert st["n_clusters_008"] == 2 and st["covered"] == 1
    assert st["vla_cluster_rank"] == 2 and abs(st["vla_cluster_share"] - 0.5) < 1e-9  # tie 32/32 -> min index
    st2, _ = analyse(P, G, P[0] + np.array([0, 0, 0.2]), G[0])
    assert st2["covered"] == 0 and abs(st2["min_dist_to_vla"] - 0.2) < 1e-9


def main():
    for f in (test_distance, test_linkage_simple, test_linkage_vs_scipy, test_candidates_and_selectors,
              test_misc, test_r1d_analyse):
        f()
        print(f"[PASS] {f.__name__}")
    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
