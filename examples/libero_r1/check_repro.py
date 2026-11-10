"""Smoke tests 9 and 10 helpers (numpy only).

  # 9: two runs of the same episode must agree on steps and success
  python -m examples.libero_r1.check_repro same results/R1/smoke/A_rep1/libero_goal results/R1/smoke/A_rep2/libero_goal
  # 10: after a kill + restart, every (task, episode) appears exactly once
  python -m examples.libero_r1.check_repro unique results/R1/smoke/resume/libero_goal --expect 4
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter


def rows(d):
    with open(os.path.join(d, "episodes.csv"), newline="") as fp:
        return list(csv.DictReader(fp))


def same(a, b):
    ra = {(r["task_id"], r["episode_id"]): r for r in rows(a)}
    rb = {(r["task_id"], r["episode_id"]): r for r in rows(b)}
    ok = bool(ra) and set(ra) == set(rb)
    for k in sorted(ra):
        if k not in rb:
            continue
        m = ra[k]["steps"] == rb[k]["steps"] and ra[k]["success"] == rb[k]["success"]
        ok &= m
        print(f"task {k[0]} ep {k[1]}: steps {ra[k]['steps']} vs {rb[k]['steps']}, "
              f"success {ra[k]['success']} vs {rb[k]['success']} -> {'same' if m else 'DIFFERENT'}")
    qa, qb = os.path.join(a, "queries.jsonl"), os.path.join(b, "queries.jsonl")
    if os.path.exists(qa) and os.path.exists(qb):
        la = [json.loads(l) for l in open(qa) if l.strip()]
        lb = [json.loads(l) for l in open(qb) if l.strip()]
        sel_a = [(q["step"], q["selected_sample_idx"], q["n_clusters"]) for q in la]
        sel_b = [(q["step"], q["selected_sample_idx"], q["n_clusters"]) for q in lb]
        print(f"query-level (step, selected, n_clusters) identical: {sel_a == sel_b} ({len(la)} vs {len(lb)} queries)")
    print("SMOKE-9", "PASS" if ok else "FAIL")
    return ok


def unique(d, expect):
    r = rows(d)
    c = Counter((x["task_id"], x["episode_id"]) for x in r)
    dup = {k: v for k, v in c.items() if v > 1}
    q = os.path.join(d, "queries.jsonl")
    qep = set()
    if os.path.exists(q):
        qep = {(str(j["task_id"]), str(j["episode_id"])) for j in map(json.loads, filter(str.strip, open(q)))}
    orphan = qep - set(c)
    ok = not dup and not orphan and (expect is None or len(c) == expect)
    print(f"episodes={len(c)} duplicates={dup} query-lines-without-episode={sorted(orphan)}")
    print("SMOKE-10", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "same":
        sys.exit(0 if same(sys.argv[2], sys.argv[3]) else 1)
    elif mode == "unique":
        exp = int(sys.argv[sys.argv.index("--expect") + 1]) if "--expect" in sys.argv else None
        sys.exit(0 if unique(sys.argv[2], exp) else 1)
    else:
        raise SystemExit(__doc__)
