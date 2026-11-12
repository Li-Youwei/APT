"""Summary tables for R1 phase 1 (spec 4.2 summary, 4.3 tables 1, 2, 4, 5, 6, and the
anomaly list of section 1). numpy only; any env, repo root:

  python -m examples.libero_r1.summarize_r1 --r1c_root results/R1/R1C \
      --r1d_csv results/R1/R1D/r1d_rows.csv --out_dir results/R1/summary

Expects <r1c_root>/<arm>/<suite>/episodes.csv (+ queries.jsonl for VA arms).
Writes CSV files and summary.md; anomalies are marked with 🔴.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from .libero_common import MAX_STEPS, SUITES
from .traj_select import wilson_interval

ARMS = ["vla", "va_oracle", "va_random"]
ARM_LABEL = {"vla": "D vla", "va_oracle": "C va_oracle", "va_random": "A va_random", "va_jev": "B va_jev"}
EXPECTED_EPISODES = 500  # 10 tasks x 50


def read_csv(path):
    with open(path, newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def md_table(rows, fields):
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(f, "")) for f in fields) + " |")
    return "\n".join(out)


def load_episodes(root, arm):
    eps = {}
    for s in SUITES:
        p = os.path.join(root, arm, s, "episodes.csv")
        if os.path.exists(p):
            eps[s] = read_csv(p)
    return eps


def iter_queries(root, arm, suite, fields):
    p = os.path.join(root, arm, suite, "queries.jsonl")
    if not os.path.exists(p):
        return
    with open(p) as fp:
        for line in fp:
            if line.strip():
                q = json.loads(line)
                yield {f: q.get(f) for f in fields}


def summarize_r1c(args, md, flags):
    data = {arm: load_episodes(args.r1c_root, arm) for arm in ARMS}
    # ── table 1: arm x suite ──
    t1, rate = [], {}
    for arm in ARMS:
        suite_rates = []
        for s in SUITES:
            rows = data[arm].get(s, [])
            n = len(rows)
            k = sum(int(r["success"]) for r in rows)
            lo, hi = wilson_interval(k, n)
            r_ = k / n if n else float("nan")
            rate[(arm, s)] = r_ if n else None
            if n:
                suite_rates.append(r_)
            t1.append(dict(arm=arm, suite=s, n=n, successes=k, rate=round(r_, 4) if n else "",
                           ci95_low=round(lo, 4) if n else "", ci95_high=round(hi, 4) if n else "",
                           complete=int(n == EXPECTED_EPISODES)))
        avg = float(np.mean(suite_rates)) if len(suite_rates) == len(SUITES) else None
        rate[(arm, "avg")] = avg
        t1.append(dict(arm=arm, suite="avg4", n="", successes="", rate=round(avg, 4) if avg is not None else "",
                       ci95_low="", ci95_high="", complete=""))
    write_csv(os.path.join(args.out_dir, "t1_success_arm_suite.csv"), t1)
    md.append("## 表 1 成功率：臂 × suite（95% Wilson 区间）\n")
    mdrows = []
    for arm in ARMS:
        r = {"臂": ARM_LABEL[arm]}
        for s in SUITES:
            row = next(x for x in t1 if x["arm"] == arm and x["suite"] == s)
            r[s] = (f"{pct(row['rate'])} [{pct(row['ci95_low'])}, {pct(row['ci95_high'])}] (n={row['n']})"
                    if row["n"] else "n/a")
        r["四 suite 平均"] = pct(rate[(arm, "avg")])
        mdrows.append(r)
    md.append(md_table(mdrows, ["臂"] + SUITES + ["四 suite 平均"]) + "\n")
    incomplete = [x for x in t1 if x["suite"] != "avg4" and x["n"] not in ("", EXPECTED_EPISODES)]
    if incomplete:
        md.append("未跑满 500 个 episode 的组合：" + ", ".join(f"{x['arm']}/{x['suite']}={x['n']}" for x in incomplete) + "\n")

    # anomalies 1-3 (section 1)
    d_avg = rate.get(("vla", "avg"))
    if d_avg is not None and d_avg < 0.90:
        flags.append(f"🔴 异常 1：D 臂四 suite 平均 {pct(d_avg)} < 90%（怀疑环境、协议或动作转换有 bug）")
    a_goal = rate.get(("va_random", "libero_goal"))
    if a_goal is not None and a_goal > 0.60:
        flags.append(f"🔴 异常 2：A 臂 Goal {pct(a_goal)} > 60%（怀疑 VA 非多模或语言泄漏）")
    for s in SUITES:
        c = rate.get(("va_oracle", s))
        if c is not None and c < 0.70:
            flags.append(f"🔴 异常 3：C 臂 {s} {pct(c)} < 70%（怀疑 VA 能力、聚类阈值或候选截断）")
    # expectations (informational)
    exp = []
    if d_avg is not None:
        exp.append(f"D 平均 ≥ 93%：{'符合' if d_avg >= 0.93 else '不符合'}（{pct(d_avg)}）")
    if a_goal is not None:
        exp.append(f"A Goal ≤ 40%：{'符合' if a_goal <= 0.40 else '不符合'}（{pct(a_goal)}）")
    for s in SUITES:
        c, d = rate.get(("va_oracle", s)), rate.get(("vla", s))
        if c is not None and d is not None:
            exp.append(f"C ≥ D − 15pp @ {s}：{'符合' if c >= d - 0.15 else '不符合'}（C {pct(c)}，D {pct(d)}）")
    if exp:
        md.append("预期核对（仅供参考）：\n\n" + "\n".join("- " + e for e in exp) + "\n")

    # ── table 2: arm x task per suite ──
    md.append("## 表 2 成功率：臂 × 任务\n")
    t2 = []
    for s in SUITES:
        tasks = sorted({int(r["task_id"]) for arm in ARMS for r in data[arm].get(s, [])})
        if not tasks:
            continue
        mdrows = []
        for t in tasks:
            r = {"task": t}
            for arm in ARMS:
                rows = [x for x in data[arm].get(s, []) if int(x["task_id"]) == t]
                k, n = sum(int(x["success"]) for x in rows), len(rows)
                r[ARM_LABEL[arm]] = f"{pct(k / n)} ({k}/{n})" if n else "n/a"
                t2.append(dict(suite=s, task_id=t, arm=arm, n=n, successes=k, rate=round(k / n, 4) if n else ""))
                if arm == "vla" and rows:
                    r["instruction"] = rows[0]["instruction"]
            mdrows.append(r)
        md.append(f"### {s}\n\n" + md_table(mdrows, ["task"] + [ARM_LABEL[a] for a in ARMS] + ["instruction"]) + "\n")
    write_csv(os.path.join(args.out_dir, "t2_success_arm_task.csv"), t2)

    # ── table 4: mean #clusters vs step (A, C) ──
    t4 = []
    for arm in ("va_random", "va_oracle"):
        for s in SUITES:
            acc = defaultdict(list)
            for q in iter_queries(args.r1c_root, arm, s, ["step", "n_clusters"]):
                acc[int(q["step"])].append(q["n_clusters"])
            for st in sorted(acc):
                t4.append(dict(arm=arm, suite=s, step=st, mean_n_clusters=round(float(np.mean(acc[st])), 4),
                               n_queries=len(acc[st])))
    write_csv(os.path.join(args.out_dir, "t4_nclusters_vs_step.csv"), t4,
              ["arm", "suite", "step", "mean_n_clusters", "n_queries"])
    md.append("## 表 4 A、C 臂平均聚类数随步数变化\n\n完整数据见 `t4_nclusters_vs_step.csv`（每 8 步一个点）。下表为每 40 步抽样：\n")
    mdrows = [r for r in t4 if r["step"] % 40 == 0]
    md.append(md_table(mdrows, ["arm", "suite", "step", "mean_n_clusters", "n_queries"]) + "\n")

    # ── table 5: C oracle_dist ──
    t5 = []
    for s in SUITES:
        succ = {(int(r["task_id"]), int(r["episode_id"])): int(r["success"]) for r in data["va_oracle"].get(s, [])}
        all_d, by = [], {1: [], 0: []}
        for q in iter_queries(args.r1c_root, "va_oracle", s, ["task_id", "episode_id", "oracle_dist"]):
            if q["oracle_dist"] is None:
                continue
            all_d.append(q["oracle_dist"])
            key = (int(q["task_id"]), int(q["episode_id"]))
            if key in succ:
                by[succ[key]].append(q["oracle_dist"])
        if not all_d:
            continue
        a = np.asarray(all_d)
        t5.append(dict(suite=s, n_queries=len(a), mean=round(float(a.mean()), 4), p50=round(float(np.percentile(a, 50)), 4),
                       p95=round(float(np.percentile(a, 95)), 4),
                       mean_success_eps=round(float(np.mean(by[1])), 4) if by[1] else "",
                       mean_failure_eps=round(float(np.mean(by[0])), 4) if by[0] else ""))
    if t5:
        write_csv(os.path.join(args.out_dir, "t5_oracle_dist.csv"), t5)
        md.append("## 表 5 C 臂 oracle_dist 分布（米）\n\n" + md_table(t5, list(t5[0].keys())) + "\n")

    # ── table 6: C - D per task ──
    t6 = []
    for s in SUITES:
        for t in range(10):
            c = [int(x["success"]) for x in data["va_oracle"].get(s, []) if int(x["task_id"]) == t]
            d = [int(x["success"]) for x in data["vla"].get(s, []) if int(x["task_id"]) == t]
            if c and d:
                t6.append(dict(suite=s, task_id=t, C_rate=round(np.mean(c), 4), D_rate=round(np.mean(d), 4),
                               C_minus_D=round(np.mean(c) - np.mean(d), 4), n_C=len(c), n_D=len(d)))
    t6.sort(key=lambda r: -r["C_minus_D"])
    if t6:
        write_csv(os.path.join(args.out_dir, "t6_C_minus_D_per_task.csv"), t6)
        md.append("## 表 6 C 臂相对 D 臂的逐任务成功率差值（从大到小）\n\n" + md_table(t6, list(t6[0].keys())) + "\n")

    # extra: decisions and switching per suite
    ex = []
    for arm in ("va_random", "va_oracle"):
        for s in SUITES:
            rows = data[arm].get(s, [])
            if not rows:
                continue
            sw = [q["dist_selected_to_prev"] for q in iter_queries(args.r1c_root, arm, s, ["dist_selected_to_prev"])
                  if q["dist_selected_to_prev"] is not None]
            ex.append(dict(arm=arm, suite=s,
                           mean_queries_per_ep=round(np.mean([int(r["n_queries"]) for r in rows]), 2),
                           mean_decisions_per_ep=round(np.mean([int(r["n_decisions"]) for r in rows]), 2),
                           mean_dist_selected_to_prev=round(float(np.mean(sw)), 4) if sw else "",
                           mean_wall_time_s=round(np.mean([float(r["wall_time_s"]) for r in rows]), 1)))
    if ex:
        write_csv(os.path.join(args.out_dir, "extra_query_stats.csv"), ex)
        md.append("## 附：每 episode 查询/决策数与相邻选择的跳变\n\n" + md_table(ex, list(ex[0].keys())) + "\n")


def summarize_r1d(args, md, flags):
    if not args.r1d_csv or not os.path.exists(args.r1d_csv):
        return
    rows = read_csv(args.r1d_csv)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["suite"], int(r["ckpt_iter"]))].append(r)
    out = []
    for (s, it) in sorted(groups, key=lambda k: (SUITES.index(k[0]), k[1])):
        g = groups[(s, it)]
        share = [float(r["vla_cluster_share"]) for r in g]
        out.append(dict(suite=s, ckpt_iter=it, n_states=len(g),
                        mean_n_clusters_004=round(np.mean([int(r["n_clusters_004"]) for r in g]), 3),
                        mean_n_clusters_008=round(np.mean([int(r["n_clusters_008"]) for r in g]), 3),
                        mean_n_clusters_012=round(np.mean([int(r["n_clusters_012"]) for r in g]), 3),
                        coverage=round(np.mean([int(r["covered"]) for r in g]), 4),
                        mean_min_dist_to_vla=round(np.mean([float(r["min_dist_to_vla"]) for r in g]), 4),
                        vla_cluster_share_mean=round(float(np.mean(share)), 4),
                        vla_cluster_share_min=round(float(np.min(share)), 4)))
        if s == "libero_goal" and np.mean([int(r["n_clusters_008"]) for r in g]) < 2:
            flags.append(f"🔴 异常 5：R1-D Goal（检查点 {it}）初始帧平均聚类数 "
                         f"{np.mean([int(r['n_clusters_008']) for r in g]):.2f} < 2")
    write_csv(os.path.join(args.out_dir, "r1d_summary.csv"), out)
    md.append("## R1-D 汇总：suite × 检查点（聚类阈值 0.08 m 为主，另列 0.04 / 0.12）\n\n"
              + md_table(out, list(out[0].keys())) + "\n")


def summarize_cap800(args, md):
    """Spec 8.8 extra check: D arm with a raised step cap vs the main D arm, same tasks."""
    cap = load_episodes(args.r1c_root, args.cap_run)
    if not cap:
        return
    base = load_episodes(args.r1c_root, "vla")
    rows = []
    for s in SUITES:
        for t in sorted({int(r["task_id"]) for r in cap.get(s, [])}):
            c = [r for r in cap[s] if int(r["task_id"]) == t]
            d = [r for r in base.get(s, []) if int(r["task_id"]) == t]
            c_succ = [int(r["steps"]) for r in c if int(r["success"])]
            rows.append(dict(
                suite=s, task_id=t, orig_cap=MAX_STEPS[s],
                D_rate=round(np.mean([int(r["success"]) for r in d]), 4) if d else "", n_D=len(d),
                cap_rate=round(np.mean([int(r["success"]) for r in c]), 4), n_cap=len(c),
                cap_successes_beyond_orig_cap=sum(st > MAX_STEPS[s] for st in c_succ),
                cap_median_steps_success=int(np.median(c_succ)) if c_succ else "",
                cap_failures_hitting_cap=sum(1 for r in c if not int(r["success"])),
            ))
    write_csv(os.path.join(args.out_dir, "extra_vla_cap800.csv"), rows)
    md.append(f"## 附加检查（规格 8.8）：D 臂提高步数上限（{args.cap_run}）\n\n"
              "`cap_successes_beyond_orig_cap` = 在原上限之后才成功的 episode 数；失败 episode 都跑满了新上限。\n\n"
              + md_table(rows, list(rows[0].keys())) + "\n")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    md, flags = ["# R1 第一阶段汇总\n"], []
    body = []
    summarize_r1d(args, body, flags)
    summarize_r1c(args, body, flags)
    summarize_cap800(args, body)
    md.append("## 异常（规格第 1 节）\n")
    md.append("\n".join("- " + f for f in flags) + "\n" if flags else "未触发任何异常条件。\n")
    md += body
    with open(os.path.join(args.out_dir, "summary.md"), "w") as fp:
        fp.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--r1c_root", default="results/R1/R1C")
    p.add_argument("--r1d_csv", default="results/R1/R1D/r1d_rows.csv")
    p.add_argument("--out_dir", default="results/R1/summary")
    p.add_argument("--cap_run", default="vla_cap800", help="run_name of the spec 8.8 extra D-arm check")
    main(p.parse_args())
