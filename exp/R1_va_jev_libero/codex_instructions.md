# R1 第一阶段：给 Codex 的操作说明（草稿）

> 状态：代码已写完，未在 GPU/LIBERO 上运行过；已按规格第 7 节裁决更新。首次运行中发现命令有误，按第 7 节回传。所有命令在仓库根目录执行。

## 0. 环境
- 代码：`git clone https://github.com/Li-Youwei/APT.git && cd APT && git checkout r1-phase1`。
- APT 环境（训练与推理服务）：Python 3.10，`pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126 && pip install -r requirements.txt`。首次建模会从 HF 拉取 `Qwen/Qwen3-VL-2B-Instruct`（必要时设 `HF_ENDPOINT`）。
- LIBERO 环境（评测、诊断、数据转换）：在已有环境中补装 `pip install Pyro4`；`export MUJOCO_GL=egl`。不需要 openvla-oft。
- 两个环境都要：`export APT_DATA_LOC_HOST=r1`。
- 离线自检（任意环境，需要 numpy，scipy 可选）：`python -m examples.libero_r1.selftest`，必须输出 `ALL SELFTESTS PASSED`。

## 1. 权重
```bash
huggingface-cli download KechunXu1/apt_models --include "apt_va/*" "apt_vla_ftlibero/*" --local-dir checkpoints/hf
python scripts/test_ckpt.py --ckpt checkpoints/hf/apt_va/ckpt_latest.pt --train-stage 0   # 冒烟 1（层数检查）
```
严格加载由训练入口完成（`--pretrained_ckpt` 走 `load_state_dict(strict=True)`，冒烟 2 启动成功即证明严格加载成功）。

## 2. 数据（服务器上没有仓库格式 LIBERO 数据时）
```bash
# LIBERO 环境，在 LIBERO 仓库内下载官方示教
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial libero_object libero_goal libero_10
# 回到 APT 仓库根目录，每个 suite 一个进程（可并行）
for s in libero_spatial libero_object libero_goal libero_10; do
  python -m examples.libero_r1.convert_libero --raw_root <LIBERO>/libero/datasets --out_root /data/libero_apt --suites $s > logs_convert_$s.txt 2>&1 &
done; wait
# APT 环境
python -m examples.libero_r1.make_data_loc --root /data/libero_apt --host r1
python -m examples.libero_r1.check_data      # 每个 suite 打印 episode 数（预期 500）和张量形状
```
默认即规格第 7 节裁决 1 的做法：动作重放、过滤空动作、丢弃重放失败的示教，不要加 `--replay states` 或 `--keep_noops`。统计结果在 `/data/libero_apt/<suite>_stats.json`（`total.n_written` = 保留数，`total.n_dropped` = 丢弃数，另有过滤掉的空动作帧数），复制到 `results/R1/R1T/data_stats/`，并在结果包说明中逐 suite 写明保留与丢弃的 episode 数。`check_data` 打印的 episode 数应等于保留数。

## 3. R1-T
冒烟（冒烟 1、2）：
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 -m apt.train --backend ddp --config r1_va_libero4 -s va_libero4_r1_smoke \
  --pretrained_ckpt checkpoints/hf/apt_va/ckpt_latest.pt --max_iterations 200 --log_interval 1 --save_interval 200 --save_latest_interval 200
python -m examples.libero_r1.r1t_report --log_dir logs/APT/va_libero4_r1_smoke --out_dir results/R1/smoke/R1T --every 1 --smoke_check
```
通过：无报错，输出 `[SMOKE-2] mean total_loss steps 1-20 = …, steps 181-200 = … -> PASS`（第 181–200 步均值低于第 1–20 步均值，规格第 7 节裁决 7）。显存不足时加 `--bs 16 --gradient_accumulation_steps 2` 并记录。

正式：
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 -m apt.train --backend ddp --config r1_va_libero4 -s va_libero4_r1 \
  --pretrained_ckpt checkpoints/hf/apt_va/ckpt_latest.pt
python -m examples.libero_r1.r1t_report --log_dir logs/APT/va_libero4_r1 --ckpt_dir checkpoints/APT/va_libero4_r1 --out_dir results/R1/R1T
```
得到 `checkpoints/APT/va_libero4_r1/ckpt_00{1,2,3,4}0000.pt`。

## 4. 推理服务（APT 环境）
```bash
pyro4-ns -p 9091 &
CUDA_VISIBLE_DEVICES=1 python -m apt.infer.remote_service --ckpt checkpoints/hf/apt_vla_ftlibero/ckpt_latest.pt --uri vla --ns_port 9091 --ensemble 0 &
CUDA_VISIBLE_DEVICES=0 python -m apt.infer.remote_service --ckpt checkpoints/APT/va_libero4_r1/ckpt_0040000.pt --uri va --ns_port 9091 --ensemble 0 &
```
VA 服务起来后（LIBERO 环境）：`python -m examples.libero_r1.check_sampling --uri va` —— [1][2] 必须 PASS；[3] 的数值写入结果包（共享 VLM 与复制观测的差异）。

## 5. 按第 6 节顺序执行
1. D 臂冒烟（冒烟 5，≥5/6）：
   `python -m examples.libero_r1.eval_r1 --arm vla --suite libero_goal --task_end 2 --episodes_per_task 3 --out_root results/R1/smoke --run_name D`
   D 臂全量：对 4 个 suite 各跑 `python -m examples.libero_r1.eval_r1 --arm vla --suite <suite>`（可并行，多个客户端可共享同一服务）。
2. R1-D 冒烟（冒烟 3、4）：VA 服务改用 smoke 检查点启动，然后
   `python -m examples.libero_r1.diag_r1d --va_ckpts $PWD/checkpoints/APT/va_libero4_r1_smoke/ckpt_0000200.pt --suites libero_goal --tasks 0 --states 0 --out_dir results/R1/smoke/R1D --vis_all`
   通过：一行 CSV + 一张图；人工看图，白黑小圆点（起点）落在夹爪上。
   冒烟 11（规格第 7 节裁决 2，紧接 R1-D 冒烟，VA 服务仍是 smoke 检查点）：
   `python -m examples.libero_r1.check_sampling --uri va --smoke11`
   通过：`[SMOKE-11] ... -> PASS`（真实指令与 `do nothing` 的 32 条轨迹位置逐元素最大差值 ≤ 1e-4 米）。不通过：停止，回传完整输出，不要继续。
   之后把 VA 服务换回 40k 检查点（第 4 节命令）。
   R1-D 全量：`python -m examples.libero_r1.diag_r1d --va_ckpts $PWD/checkpoints/APT/va_libero4_r1/ckpt_00{1,2,3,4}0000.pt --out_dir results/R1/R1D`
3. C 臂冒烟（冒烟 6，≥3/6，VA 用 40k）：
   `python -m examples.libero_r1.eval_r1 --arm va_oracle --suite libero_goal --task_end 2 --episodes_per_task 3 --out_root results/R1/smoke --run_name C`
   冒烟 9：`--arm va_random --task_end 1 --episodes_per_task 1` 分别用 `--run_name A_rep1`、`A_rep2` 跑两次，然后 `python -m examples.libero_r1.check_repro same results/R1/smoke/A_rep1/libero_goal results/R1/smoke/A_rep2/libero_goal`。
   冒烟 10：`--arm va_random --task_end 1 --episodes_per_task 2 --run_name resume --debug_kill_at_query 45`（进程会以 17 退出），去掉 `--debug_kill_at_query` 重跑，日志中应出现 `[SKIP]`，再 `python -m examples.libero_r1.check_repro unique results/R1/smoke/resume/libero_goal --expect 2`。
   C 臂全量：4 个 suite 各跑 `--arm va_oracle`。
4. A 臂全量：4 个 suite 各跑 `--arm va_random`。
5. 汇总：`python -m examples.libero_r1.summarize_r1`（输出 `results/R1/summary/summary.md`，异常标 🔴）。

中断后用同一命令重跑即可续跑。算力紧张时：D、C 优先于 A，`libero_goal` 优先。

## 6. 打包
```bash
tar --exclude='*.pt' -czf R1_phase1_results.tar.gz results/R1
```
同时附上 `nvidia-smi`、`pip freeze`（两个环境）、`git rev-parse HEAD`。

## 7. 出错时回传
完整命令、完整报错栈、服务端日志最后 200 行、`get_meta` 输出（日志中 `[INFO] VA meta`）、出错的 suite/task/episode，以及对应 `run_meta_*.json`。

## 8. 已裁决事项（见规格第 7 节，优先于规格正文）
1. 数据转换：OpenVLA 做法（动作重放、过滤空动作、丢弃重放失败的示教），报告每个 suite 保留与丢弃数。已是 `convert_libero` 默认。
2. VA 查询传真实指令；冒烟 11 检查语言泄漏（`check_sampling --smoke11`），放在 R1-D 冒烟之后。
3. 种子：每个 episode 一个种子，派生 VA、VLA 两条独立随机数流，episode 内依次取噪声；D 臂相同。
4. A 臂 `selected_candidate` 记下标 0 样本所在簇的大小排名，并记录实际执行的轨迹（`executed_pos` / `executed_gripper`）。
5. `vla_cluster_rank` = 离参照轨迹最近的样本所在簇的大小排名（从 1 起）。
6. 聚类合并条件：距离 ≤ 阈值；最大步数判定 `t >= 上限`，A、C、D 三臂一致。
7. 冒烟 2：第 181–200 步 `total_loss` 均值低于第 1–20 步均值；正式 CSV 每 1000 步一行，值为前 100 步均值。
8. 冒烟 9、10 在 C 臂冒烟之后、C 臂全量之前；冒烟 11 在 R1-D 冒烟之后。

