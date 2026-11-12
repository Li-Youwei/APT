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
VA 服务起来后（LIBERO 环境）：`python -m examples.libero_r1.check_sampling --uri va` —— 两个参考相机（agentview、eye_in_hand）各自的 [1][2] 必须 PASS；[3] 只记录数值，WARN 不算失败（规格 8.4）；[4] 只是诊断输出（同一噪声在两个参考相机下的差异，预期很大）。

**推理配置固定（规格 8.5）**：所有臂、所有 episode 都用共享骨干（`share_vlm=True`，代码默认，命令行不可改）、bf16（服务端固定）、VA 每次查询 K=32（`traj_select.K_SAMPLES`）。不要改服务端代码、不要换服务启动参数；任何一项中途变化，受影响的 suite 整体重跑。

> 2026-11-11 修复：动作参考相机原先由服务端用未设种子的 `random` 每次随机挑选，导致同一噪声两次调用结果不同（检查 [1] 失败）。现在由客户端从 episode 种子派生的独立随机流抽取并显式传给服务，分布不变（两相机各 1/2），噪声流不变。改了服务端代码，必须重启两个推理服务后再复验；客户端若连到旧服务会直接报 `server ignored ref_cam`。

## 5B. 2026-11-12 诊断实验 R1-X（规格第 9 节）—— 现在从这里开始

规格 9.1：C 臂全量、A 臂全量、800 步附加检查全部暂停；R1-D 结果作废（保留文件但不再使用）。5A 第 2 步之后的内容暂停，等研究侧根据 R1-X 结果另行通知。R1-X 只做推理，不训练，不改权重。

0. `git pull`（分支 `r1-phase1`），重启 pyro4-ns、VLA 服务（GPU1）和 VA 服务（GPU0，用 40k 启动即可，脚本会在 `apt_va`、10k、40k 之间热切换）。服务端代码有新增方法（`sample_trajs_legacy`），必须重启。先建输出目录：`mkdir -p results/R1/R1X/smoke`。
1. **X1 冒烟（规格 9.4 第 1、2 条）**，LIBERO 环境：
   ```bash
   python -u -m examples.libero_r1.diag_r1x --tasks 0 --ref_cams agentview --out_dir results/R1/R1X/smoke 2>&1 | tee results/R1/R1X/smoke_x1.log
   ```
   通过：`results/R1/R1X/smoke/x1_rows.csv` 有 2 行（VLA × 共享骨干 / 复制观测），并输出 `[X1-SMOKE-2] ... -> PASS`（VLA 相邻步位移中位数 ≤ 1 cm）。不通过：停止并回传。
2. **X1 全量**（4 模型 × 2 路径 × 2 相机 × 10 状态 = 160 行）：
   ```bash
   python -u -m examples.libero_r1.diag_r1x \
     --va_ckpts apt_va=$PWD/checkpoints/hf/apt_va/ckpt_latest.pt \
                va10k=$PWD/checkpoints/APT/va_libero4_r1/ckpt_0010000.pt \
                va40k=$PWD/checkpoints/APT/va_libero4_r1/ckpt_0040000.pt \
     --out_dir results/R1/R1X 2>&1 | tee results/R1/R1X/x1.log
   ```
   输出 `x1_rows.csv`、`x1_summary.csv`、`x1_vis/va40k_goal_t*.png`（10 张，白线为 VLA 参照）。可续跑。
3. 停掉两个推理服务，释放 GPU。
4. **X2 冒烟（规格 9.4 第 3 条）**，APT 环境（需 `APT_DATA_LOC_HOST=r1`）：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1x_x2 \
     --models vla=checkpoints/hf/apt_vla_ftlibero/ckpt_latest.pt \
     --out_dir results/R1/R1X/smoke 2>&1 | tee results/R1/R1X/smoke_x2.log
   ```
   通过：`[X2-SMOKE-3] vla mean 32-step position error = … cm (<= 3 cm) -> PASS`。不通过：停止并回传（说明数据读取或转换与 VLA 训练数据不一致）。
5. **X2 全量**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1x_x2 \
     --models vla=checkpoints/hf/apt_vla_ftlibero/ckpt_latest.pt \
              apt_va=checkpoints/hf/apt_va/ckpt_latest.pt \
              va40k=checkpoints/APT/va_libero4_r1/ckpt_0040000.pt \
     --out_dir results/R1/R1X 2>&1 | tee results/R1/R1X/x2.log
   ```
   输出 `x2_gt.csv`（真值统计）、`x2_pred.csv`（逐样本预测误差与平滑度）、`x2_loss.csv`（逐样本 × 时间步档的 loss）、`x2_summary.json`。`x2_samples.pt` 是 200 个样本的缓存，不打包。
6. **回传**：
   ```bash
   tar --exclude='*.pt' -czf R1X_results.tar.gz results/R1/R1X
   ```
   另附：两个服务的完整日志、`git rev-parse HEAD`、`nvidia-smi`。X3 的书面回答在仓库 `exp/R1_va_jev_libero/r1x_x3_answers.md`，不需要 Codex 处理。

R1-X 只新增诊断入口（`sample_trajs_legacy`、`ActionExpert.forward` 的可选 `train_timesteps`），没有改变任何已有推理或训练路径的行为，满足规格 9 节“在未修改的代码上留证据”的要求。

## 5A. 2026-11-12 恢复执行（规格第 8 节第 7 条）（第 2 步之后暂停，见 5B）

已完成且保留、不重跑：R1-T、D 臂全量（平均 91.75%，规格 8.6 接受为参照系）、R1-D 冒烟（冒烟 3、4）。

0. 更新代码并重启服务：`git pull`（分支 `r1-phase1`），然后按第 4 节命令重启 pyro4-ns、VLA 服务（GPU1）和 **40k** VA 服务（GPU0）。以下命令都在 LIBERO 环境、仓库根目录执行，环境变量同第 0 节。
1. **冒烟 11（40k，规格 8.1–8.3）**，两条命令都必须退出码 0：
   ```bash
   python -u -m examples.libero_r1.check_sampling --uri va  --smoke11
   python -u -m examples.libero_r1.check_sampling --uri vla --smoke11
   ```
   判据：对照指令为补到与真实指令 token 数相同的 `do nothing now now …`；VA 侧每个参考相机位置最大差 ≤ 1e-4 m，VLA 侧（阳性对照）每个参考相机 > 0.01 m。输出 `[SMOKE-11] overall (VA side): PASS` 与 `overall (VLA side): PASS`。VA 侧失败：停止并回传。日志存 `results/R1/smoke/smoke11_40k_{va,vla}.log`。
2. **R1-D 全量**：
   ```bash
   python -u -m examples.libero_r1.diag_r1d --va_ckpts $PWD/checkpoints/APT/va_libero4_r1/ckpt_00{1,2,3,4}0000.pt --out_dir results/R1/R1D
   ```
   结束时 VA 服务停在最后加载的 40k 权重上，可直接继续。
3. **C 臂冒烟（冒烟 6，≥3/6）**：
   ```bash
   python -u -m examples.libero_r1.eval_r1 --arm va_oracle --suite libero_goal --task_end 2 --episodes_per_task 3 --out_root results/R1/smoke --run_name C
   ```
4. **冒烟 9（A 臂可复现）与冒烟 10（断点续跑）**：
   ```bash
   python -u -m examples.libero_r1.eval_r1 --arm va_random --suite libero_goal --task_end 1 --episodes_per_task 1 --out_root results/R1/smoke --run_name A_rep1
   python -u -m examples.libero_r1.eval_r1 --arm va_random --suite libero_goal --task_end 1 --episodes_per_task 1 --out_root results/R1/smoke --run_name A_rep2
   python -m examples.libero_r1.check_repro same results/R1/smoke/A_rep1/libero_goal results/R1/smoke/A_rep2/libero_goal
   python -u -m examples.libero_r1.eval_r1 --arm va_random --suite libero_goal --task_end 1 --episodes_per_task 2 --out_root results/R1/smoke --run_name resume --debug_kill_at_query 45   # 预期退出码 17
   python -u -m examples.libero_r1.eval_r1 --arm va_random --suite libero_goal --task_end 1 --episodes_per_task 2 --out_root results/R1/smoke --run_name resume   # 日志应有 [SKIP]
   python -m examples.libero_r1.check_repro unique results/R1/smoke/resume/libero_goal --expect 2
   ```
   通过：`SMOKE-9 PASS`、`SMOKE-10 PASS`。
5. **C 臂全量**：`libero_goal` 优先，4 个 suite 各一条（可并行，多个客户端共享两个服务）：
   ```bash
   python -u -m examples.libero_r1.eval_r1 --arm va_oracle --suite <suite>
   ```
6. **A 臂全量**：
   ```bash
   python -u -m examples.libero_r1.eval_r1 --arm va_random --suite <suite>
   ```
7. **附加检查（规格 8.8，不阻塞主流程）**：只需 VLA 服务。
   ```bash
   python -u -m examples.libero_r1.eval_r1 --arm vla --suite libero_goal --tasks 0 3 7 --max_steps_override 800 --run_name vla_cap800
   python -u -m examples.libero_r1.eval_r1 --arm vla --suite libero_10   --tasks 8     --max_steps_override 800 --run_name vla_cap800
   ```
   输出在 `results/R1/R1C/vla_cap800/<suite>/`，字段与 D 臂相同；`run_meta_*.json` 记录 `max_steps: 800`。
8. **汇总与打包**：`python -m examples.libero_r1.summarize_r1`（含附加检查表 `extra_vla_cap800.csv`），然后按第 6 节打包。

## 5. 按第 6 节顺序执行（历史记录：首次执行用，冒烟 11 以 5A 为准）
1. D 臂冒烟（冒烟 5，≥5/6）：
   `python -m examples.libero_r1.eval_r1 --arm vla --suite libero_goal --task_end 2 --episodes_per_task 3 --out_root results/R1/smoke --run_name D`
   D 臂全量：对 4 个 suite 各跑 `python -m examples.libero_r1.eval_r1 --arm vla --suite <suite>`（可并行，多个客户端可共享同一服务）。
2. R1-D 冒烟（冒烟 3、4）：VA 服务改用 smoke 检查点启动，然后
   `python -m examples.libero_r1.diag_r1d --va_ckpts $PWD/checkpoints/APT/va_libero4_r1_smoke/ckpt_0000200.pt --suites libero_goal --tasks 0 --states 0 --out_dir results/R1/smoke/R1D --vis_all`
   通过：一行 CSV + 一张图；人工看图，白黑小圆点（起点）落在夹爪上。
   冒烟 11：已由规格第 8 节改为 token 数对齐的判据并在 40k 上执行，见 5A 第 1 步。
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

## 9. 已裁决事项（规格第 8 节，优先于第 7 节和正文）
1. 冒烟 11 判定通过、VA 无语言泄漏（token 数对齐时 VA 逐位相同，VLA 差 129–150 mm）。
2. 冒烟 11 新判据：对照补到相同 token 数；VA 位置最大差 ≤ 1e-4 m；VLA 阳性对照 > 0.01 m（`check_sampling --smoke11`，两个服务各跑一次）。
3. 冒烟 11 在 40k 上重跑一次，排在 R1-D 全量之前；VA 侧失败则停止上报。
4. 检查 [3] 的 WARN 不视为 bug；正式运行统一用共享骨干；可复现性由冒烟 9 保证。
5. 所有臂、所有 episode 用同一套推理配置（共享骨干、bf16、K=32），中途改变则受影响 suite 整体重跑。
6. D 臂 91.75% 接受为参照系，不重跑。
7. 恢复执行顺序见 5A。
8. 附加检查：D 臂在 `libero_goal` 任务 0、3、7 与 `libero_10` 任务 8 各 50 个 episode，上限 800 步，输出 `results/R1/R1C/vla_cap800/`。

## 10. 规格第 9 节要点
1. C 臂冒烟 0/6 判定为待查的 bug 或训练失败，不是 VA 能力问题。
2. C 臂全量、A 臂全量、800 步附加检查暂停；R1-D 结果作废，修复后整体重跑。
3. 冒烟 4 记为通过（VLA 参照轨迹起点落在夹爪上）。
4. 诊断实验 R1-X：X1 推理路径对照、X2 训练数据开环对照、X3 代码核对（书面回答见 `r1x_x3_answers.md`）。执行见 5B。

