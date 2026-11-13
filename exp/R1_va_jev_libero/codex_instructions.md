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

## 5D. 诊断实验 R1-Z（规格第 13 节）—— 现在从这里开始

C 臂、A 臂、R1-D 继续暂停。新数据（带指令目标）是今后唯一的训练数据。**本节所有命令都在设置了新数据 YAML 的 shell 中执行**（APT 环境）：
```bash
export APT_DATA_LOC=$PWD/data_utils/data_loc.r1b.local.yaml APT_DATA_LOC_HOST=r1
mkdir -p results/R1/R1Z/smoke
```
只推理部分（13.2）占 GPU0，过拟合训练（13.3）占 GPU1，两边可并行。不需要推理服务。完整输出都用 `2>&1 | tee results/R1/R1Z/<步骤名>.log` 保存。先 `git pull`（分支 `r1-phase1`）。

运行 Q 按规格第 14 节从预训练 VLA `apt_vla` 直接加载（`train_stage=1`，不走 `load_from_va`），先下载（约 5.1 GB；`apt_vla` 的 VLM 是全量微调过的，权重文件里带 `vlm_weights`）：
```bash
huggingface-cli download KechunXu1/apt_models --include "apt_vla/*" --local-dir checkpoints/hf
python scripts/test_ckpt.py --ckpt checkpoints/hf/apt_vla/ckpt_latest.pt --train-stage 1 2>&1 | tee results/R1/R1Z/test_ckpt_apt_vla.log
```
`test_ckpt.py` 应输出 `[ OK ]` 并提示含 `vlm_weights`。

**冒烟（规格 13.5）**
1. Z1–Z4 各在 5 个样本上跑通（GPU0）：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1z suites --n_samples 5 --out_dir results/R1/R1Z/smoke 2>&1 | tee results/R1/R1Z/smoke_suites.log
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1z timesteps --n_samples 5 --out_dir results/R1/R1Z/smoke 2>&1 | tee results/R1/R1Z/smoke_timesteps.log
   ```
   通过：两个命令退出码 0，`z_suites_rows.csv`（含 `raw` 与 `smooth7` 两种行）和 `z2_summary.json` 有记录；并输出 `[Z-SMOKE-2] ... -> PASS`（Goal 上 VLA 最优轨迹误差中位数 ≤ 3 cm）。
2. P、Q 各训练 100 步（GPU1，依次执行；`--master_port` 必须与其他训练不同）：
   ```bash
   CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port 29511 -m apt.train --backend ddp --config overfit_s0_goal1 \
     -s overfit_s0_goal1_smoke --pretrained_ckpt checkpoints/hf/apt_va/ckpt_latest.pt \
     --max_iterations 100 --save_interval 100 --save_latest_interval 100 2>&1 | tee results/R1/R1Z/smoke_train_P.log
   CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port 29512 -m apt.train --backend ddp --config overfit_s1_goal1 \
     -s overfit_s1_goal1_smoke --pretrained_ckpt checkpoints/hf/apt_vla/ckpt_latest.pt \
     --max_iterations 100 --save_interval 100 --save_latest_interval 100 2>&1 | tee results/R1/R1Z/smoke_train_Q.log
   grep -n "Loaded VLM weights\|Loaded pretrained weights\|load_from_va" results/R1/R1Z/smoke_train_Q.log
   ```
   通过（规格 13.5 第 3 条按第 14 节修改）：两次都无报错跑完 100 步；日志开头 `num samples of LiberoGoalTask1` 为任务 1 保留的示教数（非 0）；Q 严格加载 `apt_vla` 成功——动作专家用严格模式加载（不匹配会直接报错退出），日志有 `Loaded pretrained weights from checkpoints/hf/apt_vla/ckpt_latest.pt` 和 `Loaded VLM weights from checkpoint (missing=…, unexpected=0)`，且没有 `load_from_va layer mapping` 行（说明没有走扩展流程）。
   任一冒烟不通过：停止并回传。

**正式运行**
3. 只推理部分（GPU0，各 200 个样本）：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1z suites --out_dir results/R1/R1Z 2>&1 | tee results/R1/R1Z/suites.log
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1z timesteps --out_dir results/R1/R1Z 2>&1 | tee results/R1/R1Z/timesteps.log
   ```
   `suites` 输出 Z1（Goal 上真值与 VLA 的参照值）、Z3（四个 suite 上 VA 与 VLA 的指标与聚类数）和 Z4（VA 平滑后重算的指标，`va_smooth7`），写入 `z_suites_summary.json` 和 `z_suites_rows.csv`；`timesteps` 输出 Z2，写入 `z2_summary.json`。VA 默认用 `va_libero4_r1b` 60k，VLA 用 `apt_vla_ftlibero`。
4. 过拟合训练（GPU1，P、Q 依次执行；显存不足时两者都改为 `--bs 32 --gradient_accumulation_steps 2` 并注明）：
   ```bash
   CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port 29511 -m apt.train --backend ddp --config overfit_s0_goal1 \
     -s overfit_s0_goal1 --pretrained_ckpt checkpoints/hf/apt_va/ckpt_latest.pt 2>&1 | tee results/R1/R1Z/train_P.log
   CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port 29512 -m apt.train --backend ddp --config overfit_s1_goal1 \
     -s overfit_s1_goal1 --pretrained_ckpt checkpoints/hf/apt_vla/ckpt_latest.pt 2>&1 | tee results/R1/R1Z/train_Q.log
   python -m examples.libero_r1.r1t_report --log_dir logs/APT/overfit_s0_goal1 --ckpt_dir checkpoints/APT/overfit_s0_goal1 --out_dir results/R1/R1Z/train_P --every 100
   python -m examples.libero_r1.r1t_report --log_dir logs/APT/overfit_s1_goal1 --ckpt_dir checkpoints/APT/overfit_s1_goal1 --out_dir results/R1/R1Z/train_Q --every 100
   ```
5. 过拟合训练后检查（任一空闲 GPU）：
   ```bash
   C=checkpoints/APT
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1z overfit --out_dir results/R1/R1Z/overfit --models \
     p2k=$C/overfit_s0_goal1/ckpt_0002000.pt p4k=$C/overfit_s0_goal1/ckpt_0004000.pt \
     p6k=$C/overfit_s0_goal1/ckpt_0006000.pt p8k=$C/overfit_s0_goal1/ckpt_0008000.pt \
     q2k=$C/overfit_s1_goal1/ckpt_0002000.pt q4k=$C/overfit_s1_goal1/ckpt_0004000.pt \
     q6k=$C/overfit_s1_goal1/ckpt_0006000.pt q8k=$C/overfit_s1_goal1/ckpt_0008000.pt 2>&1 | tee results/R1/R1Z/overfit.log
   ```
   100 个样本取自任务 1 的训练数据（种子 0），每个检查点采 8 条（Stage-1 用样本的真实指令）；输出 `overfit_summary.json`（含逐时间步反推误差 `by_t`）。
   注意：Q 训练时 VLM 冻结，用的是 `apt_vla` 里的 VLM 权重，但 Q 自己的检查点不保存 VLM 权重。检查脚本会读取训练配置里的 `pretrained_ckpt`，把同一份 VLM 权重加载回去；日志应有 `[OVERFIT] ...: VLM weights taken from its pretrained_ckpt checkpoints/hf/apt_vla/ckpt_latest.pt (missing=…, unexpected=0)`，`overfit_summary.json` 中 Q 各检查点的 `vlm_source` 应为该路径，P 为 `base Qwen3-VL`。
6. **完成后停止并回传**（规格 13.5：不自动进入后续步骤）：
   ```bash
   tar --exclude='*.pt' -czf R1Z_results.tar.gz results/R1/R1Z
   ```
   另附 `git rev-parse HEAD`、`nvidia-smi`。

## 5C. R1-Y 验证与 VA 重训（规格第 12 节）（第 10 步停止，转 5D）

C 臂全量、A 臂全量、800 步附加检查继续暂停。约定：`RAW=<LIBERO 仓库>/libero/datasets`（官方示教原始文件所在目录，与第 2 节转换时的 `--raw_root` 相同）。0–4 步只做推理，不训练。
每一步的完整输出都用 `2>&1 | tee results/R1/R1Y/<步骤名>.log` 保存。

0. `git pull`（分支 `r1-phase1`），`mkdir -p results/R1/R1Y`。本节不需要推理服务。
1. **控制器配置与指令目标公式核对**（LIBERO 环境）：
   ```bash
   python -u -m examples.libero_r1.controller_probe --raw_root $RAW --out results/R1/R1Y/controller_config.json
   ```
   回报 `controller` 中的 `output_max/output_min/action_scale`（预期位置 0.05 m、姿态 0.5 rad）、`position_limits/orientation_limits`，以及公式与控制器实际目标的差（`formula_vs_controller_*`，预期位置 < 1e-6 m）。这一步不判停，但数值必须回传。
2. **Y1**（APT 环境，GPU）：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1y y1 --raw_root $RAW \
     --controller_json results/R1/R1Y/controller_config.json --out_dir results/R1/R1Y
   ```
   判定行：`[Y1-VERDICT] ... -> A HOLDS / A NOT CONFIRMED`（第 1 步误差中位数 < 1 cm）。
3. **Y2**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1y y2 \
     --va checkpoints/APT/va_libero4_r1/ckpt_0040000.pt --out_dir results/R1/R1Y
   ```
   判定行：`[Y2-VERDICT] ... -> ROTATION MISMATCH (stop, report) / not triggered`。
4. **Y3**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1y y3 --raw_root $RAW \
     --controller_json results/R1/R1Y/controller_config.json --out_dir results/R1/R1Y \
     --models va10k=checkpoints/APT/va_libero4_r1/ckpt_0010000.pt va20k=checkpoints/APT/va_libero4_r1/ckpt_0020000.pt \
              va30k=checkpoints/APT/va_libero4_r1/ckpt_0030000.pt va40k=checkpoints/APT/va_libero4_r1/ckpt_0040000.pt \
              vla=checkpoints/hf/apt_vla_ftlibero/ckpt_latest.pt
   ```
   不判停，输出 `y3_summary.json`（各档由噪声预测反推的干净轨迹的位置误差，米）。
5. **是否进入重训（规格 12.3 执行条件）**：只有 Y1 为 `A HOLDS` 且 Y2 为 `not triggered` 才继续第 6 步，不需要再等研究侧通知；否则停止，打包 `results/R1/R1Y` 回传。
6. **重新转换四个 suite**（LIBERO 环境，新目录，旧数据保留）：
   ```bash
   for s in libero_spatial libero_object libero_goal libero_10; do
     python -u -m examples.libero_r1.convert_libero --raw_root $RAW --out_root /data/libero_apt_r1b --suites $s --write_desired \
       > results/R1/R1Y/convert_r1b_$s.log 2>&1 &
   done; wait
   mkdir -p results/R1/R1Tb/data_stats && cp /data/libero_apt_r1b/*_stats.json results/R1/R1Tb/data_stats/
   ```
   回报每个 suite 的保留与丢弃 episode 数（应与旧数据相同：共 1702 / 2000），以及 `desired_check`（公式与控制器的差、指令目标领先实际位置的中位数和 95 分位）。`/data/libero_apt_r1b` 换成实际磁盘位置即可。
7. **登记新数据并检查**（APT 环境）：
   ```bash
   python -m examples.libero_r1.make_data_loc --root /data/libero_apt_r1b --host r1 --out data_utils/data_loc.r1b.local.yaml
   export APT_DATA_LOC=$PWD/data_utils/data_loc.r1b.local.yaml APT_DATA_LOC_HOST=r1
   python -m examples.libero_r1.check_data
   ```
   **之后所有训练和数据相关命令都必须在设置了这个 `APT_DATA_LOC` 的 shell 中执行。** 推理服务不读数据，两套 YAML 都可以。
8. **数据检查（规格 12.3 第 2 条）**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1y datacheck --out_dir results/R1/R1Y/datacheck
   ```
   必须输出 `[DATACHECK] ... -> PASS`（VLA 对新数据真值第 1 步误差中位数 < 1 cm 且夹爪平均绝对差中位数 < 0.1），否则停止并回传。
9. **重训**（规格 12.3 第 3 条：与 2.1 相同，只改 60000 步和保存名）：
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 -m apt.train --backend ddp --config r1b_va_libero4 -s va_libero4_r1b \
     --pretrained_ckpt checkpoints/hf/apt_va/ckpt_latest.pt 2>&1 | tee results/R1/R1Y/train_r1b.log
   python -m examples.libero_r1.r1t_report --log_dir logs/APT/va_libero4_r1b --ckpt_dir checkpoints/APT/va_libero4_r1b --out_dir results/R1/R1Tb
   ```
   显存不足时同 2.1 的降批量方案并注明。得到 `ckpt_00{1..6}0000.pt`。
10. **训练后检查与选检查点（规格 12.3 第 4 条）**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1y posttrain --out_dir results/R1/R1Y/posttrain \
     --models va40k=checkpoints/APT/va_libero4_r1b/ckpt_0040000.pt va60k=checkpoints/APT/va_libero4_r1b/ckpt_0060000.pt
   ```
   通过判据：相邻步位移中位数 ≤ 1.5 cm 且 8 条中最优者的位置误差中位数 ≤ 4 cm；60k 优先，否则 40k。选中的路径写在 `results/R1/R1Y/posttrain/selected_ckpt.txt`；输出 `[POSTTRAIN-SELECT] no checkpoint passes` 时停止并回传（含各档误差）。
11. **归档旧 VA 的结果**：
    ```bash
    mkdir -p results/R1/archive_actual_pose
    mv results/R1/R1D results/R1/archive_actual_pose/R1D
    mv results/R1/smoke/C results/R1/archive_actual_pose/smoke_C
    ```
12. **恢复第一阶段（规格 12.3 第 5 条）**：`VA_CKPT=$(cat results/R1/R1Y/posttrain/selected_ckpt.txt)`，按 5A 的顺序执行，所有写“40k 检查点”的地方改用 `$VA_CKPT`：
    1. 用 `$VA_CKPT` 重启 VA 服务（VLA 服务不变）。
    2. 冒烟 11（VA 与 VLA 两侧）。
    3. R1-D 全量：`--va_ckpts` 用 `va_libero4_r1b` 的 10k、20k、30k、40k（规格 2.2 原文是四个检查点）；若选中的是 60k，再追加 60k，并加 `--vis_iters 60000`。
    4. C 臂冒烟。
    5. 冒烟 9、10（输出目录先清理旧的 `results/R1/smoke/A_rep*` 与 `resume`）。
    6. C 臂全量。
    7. A 臂全量。
    8. 800 步附加检查（D 臂结果与 VLA 无关，照常）。
    9. 汇总打包。

回传：`results/R1/R1Y/`、`results/R1/R1Tb/`、各步日志、`git rev-parse HEAD`。

## 5B. 2026-11-12 诊断实验 R1-X（规格第 9 节）（已完成）

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
   **已完成（2026-11-12），不重跑。** 冒烟第 2 条已由规格第 10 节第 1 条改为：VLA 每行相邻步位移“95 分位 ÷ 中位数”≤ 2.0 且 0.08 m 下聚类数 ≤ 3。代码更新后在 LIBERO 环境用已有记录重新判定一次（不需要服务，只读写 `x1_summary.csv`），日志存下来：
   ```bash
   python -u -m examples.libero_r1.diag_r1x --summarize_only --out_dir results/R1/R1X 2>&1 | tee results/R1/R1X/x1_resummarize.log
   ```
   预期 `[X1-SMOKE-2] VLA rows: worst p95/median = 1.79 (<= 2.0), max clusters = 2 (<= 3) over 40 rows -> PASS`。
3. 停掉两个推理服务，释放 GPU。
4. **X2 冒烟（规格 9.4 第 3 条）**，APT 环境（需 `APT_DATA_LOC_HOST=r1`）：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1x_x2 \
     --models vla=checkpoints/hf/apt_vla_ftlibero/ckpt_latest.pt \
     --out_dir results/R1/R1X/smoke 2>&1 | tee results/R1/R1X/smoke_x2.log
   ```
   **已完成（2026-11-12），不重跑。** 规格第 11 节第 1 条改为：VLA 平均位置误差的中位数 ≤ 4 cm，且预测与真值 32 步净位移中位数之比在 0.7–1.4（实测 2.85 cm、0.90，记为通过）。`results/R1/R1X/smoke/` 保留原样。
5. **X2 全量**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1x_x2 \
     --models vla=checkpoints/hf/apt_vla_ftlibero/ckpt_latest.pt \
              apt_va=checkpoints/hf/apt_va/ckpt_latest.pt \
              va40k=checkpoints/APT/va_libero4_r1/ckpt_0040000.pt \
     --out_dir results/R1/R1X 2>&1 | tee results/R1/R1X/x2.log
   ```
   先 `git pull`，代码须包含规格第 11 节第 3 条的新增记录（`git log` 中应有 "spec section 11" 的提交）。
   输出 `x2_gt.csv`（真值统计）、`x2_pred.csv`（逐样本预测误差、平滑度、第 1/2/4/8/16/32 步位置误差、夹爪平均绝对差与均值）、`x2_loss.csv`（逐样本 × 时间步档的 loss）、`x2_summary.json`。`x2_samples.pt` 是 200 个样本的缓存，不打包。
   运行结束时会按新判据打印 `[X2-SMOKE-3] vla: median … cm (<= 4 cm), net displacement ratio pred/gt = … (0.7-1.4) -> PASS/FAIL`，并打印 VLA 的逐步误差与夹爪误差。若为 FAIL：不跑 X2b，停止并回传。
6. **X2b（规格第 10 节第 4 条）**，X2 全量之后、同一 `--out_dir`（复用 `x2_samples.pt` 缓存，样本和噪声种子与 X2 相同），只跑两个 VA：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u -m examples.libero_r1.diag_r1x_x2 --x2b \
     --models apt_va=checkpoints/hf/apt_va/ckpt_latest.pt \
              va40k=checkpoints/APT/va_libero4_r1/ckpt_0040000.pt \
     --out_dir results/R1/R1X 2>&1 | tee results/R1/R1X/x2b.log
   ```
   8 组 = 2 模型 × 扩散推理步数 {20, 100} × 动作专家精度 {bf16, fp32}，每样本 8 条。输出 `x2b_pred.csv`（每模型 × 设置 × 样本一行，含与 X2 相同的逐步误差和夹爪误差字段）和 `x2b_summary.json`。冻结的 VLM 始终按训练时的 bf16 运行，精度设置只作用于动作专家。若 fp32 组报错，保留 bf16 组结果并回传完整报错栈。
7. **回传**：
   ```bash
   tar --exclude='*.pt' -czf R1X_results.tar.gz results/R1/R1X
   ```
   另附：两个服务的完整日志、`git rev-parse HEAD`、`nvidia-smi`。X3 的书面回答在仓库 `exp/R1_va_jev_libero/r1x_x3_answers.md`，不需要 Codex 处理。

R1-X 只新增诊断入口（`sample_trajs_legacy`、`ActionExpert.forward` 的可选 `train_timesteps`，X2b 的推理步数和精度只在诊断脚本里对已加载的模型临时设置），没有改变任何已有推理或训练路径的行为，满足规格 9 节“在未修改的代码上留证据”的要求。规格第 10 节第 5 条：复制观测路径下 VLA 的单步巨大跳变本轮不查，`x1_rows.csv` 原样保留。

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

## 11. 规格第 10 节要点
1. X1 冒烟第 2 条改为“95 分位 ÷ 中位数 ≤ 2.0 且聚类数 ≤ 3”，X1 记为通过，不重跑（5B 第 2 步的重新判定只是留记录）。
2. 放行 X2，按 9.2 节原样执行（5B 第 4、5 步）。
3. 新增 X2b：`apt_va` 与 40k VA × 推理步数 {20, 100} × 精度 {bf16, fp32}（5B 第 6 步）。
4. 复制观测路径下 VLA 的单步巨大跳变本轮不查，正式运行不用该路径。

## 12. 规格第 11 节要点
1. X2 冒烟第 3 条改为：VLA 平均位置误差中位数 ≤ 4 cm，且预测/真值净位移中位数之比 0.7–1.4；冒烟记为通过，不重跑。
2. 放行 X2 全量和 X2b（5B 第 5、6 步）。
3. X2 和 X2b 新增记录：第 1、2、4、8、16、32 步位置误差中位数（VA 取 8 条中平均误差最小的那条）；夹爪开合度平均绝对差中位数，以及真值、预测各自的均值。净位移统一从查询时末端位置量起。

## 13. 规格第 12 节要点
1. 推断 A（VLA 学的是控制器指令目标）基本成立；VA 更可能是在低噪声段没学好；未微调的 `apt_va` 在 LIBERO 上的表现不作判读。
2. R1-Y：Y1 用原始动作重建指令目标与 VLA 对比；Y2 比较旋转增强开和关；Y3 给出由噪声预测反推的位置误差分档表（5C 第 1–4 步）。
3. Y1 成立且 Y2 未触发时，直接重新转换（写入 `ee_pose_desired` / `gripper_desired`）→ 数据检查 → 重训 60k（`va_libero4_r1b`）→ 按 40k/60k 判据选检查点 → 恢复第一阶段（5C 第 5–12 步）。
4. 指令目标按控制器的实际实现计算（见下节），结果中报告控制器配置的实际值。

## 14. 指令目标：规格 12.2 写法与 robosuite v1.4.1 OSC 实际实现的差异
按控制器实际实现处理（`examples/libero_r1/libero_common.py` 中 `commanded_target_formula`；转换时直接读取控制器的目标）：
1. **动作先裁剪到输入范围 [-1, 1]，再乘缩放系数**（规格写法没有裁剪）。
2. **姿态目标只在旋转分量不全为 0 时才更新**；三个旋转分量都为 0 时，控制器沿用上一步的姿态目标，而不是“当前姿态”。
3. **控制器的姿态目标是夹爪 site 的姿态**，我们的末端姿态取自 hand body（`robot0_eef_quat`），两者相差一个固定旋转。左乘的旋转增量与这个固定旋转无关；转换时用每帧实测的 site→body 偏移换算，并报告其漂移。位置两者是同一个点（grip site），无差异。
4. 位置/姿态限位（`position_limits` / `orientation_limits`）在配置了时会裁剪目标，LIBERO 默认未配置，以 `controller_config.json` 的实际值为准。
5. 夹爪目标按规格：动作第 7 维 −1 → 1（张开），+1 → 0（闭合），中间值线性映射为 (1 − a)/2。
转换脚本 `--write_desired` 写入的是控制器实际目标（`controller.goal_pos` / `goal_ori`），同时计算规格公式（含上面第 1、2 条）并在 `desired_check` 中报告二者之差；Y1 / Y3 中 VLA 的“指令目标真值”无法读控制器（旧数据），用的是含第 1、2 条的公式，每个 episode 开头若第一个动作没有旋转分量，以当前姿态代替控制器复位时的姿态目标（只影响姿态，不影响 Y1 的位置判据）。

## 15. 规格第 13 节要点
1. 新数据是今后唯一的训练数据；“加步数”的判断作废；C 臂、A 臂、R1-D 继续暂停。
2. R1-Z 只推理部分：Z1 参照值、Z2 固定时间步 loss 与反推误差、Z3 分 suite 检查（含聚类数）、Z4 七步居中滑动平均后重算（只作用于 VA 采样，不改默认路径）。
3. 过拟合测试：Goal 任务 1，P（Stage-0，`apt_va` 直接加载）与 Q（Stage-1，按第 14 节改为 `apt_vla` 直接加载，不走 `load_from_va`），各 8000 步、单卡 bs 64，每 2000 步一个检查点。
4. 执行见 5D；完成后停止回传。

## 16. 规格第 14 节要点
1. 运行 Q 的初始化改为 Hugging Face `apt_vla/ckpt_latest.pt`（预训练 VLA），`--pretrained_ckpt` 直接加载，`train_stage=1`，`load_from_va=False`；保存名仍为 `overfit_s1_goal1`。不再运行 `load_from_va` 版本。
2. 冒烟第 3 条改为：Q 严格加载 `apt_vla` 成功。
3. `apt_vla` 的 VLM 是全量微调过的：Q 训练时冻结的 VLM 用的是 `apt_vla` 的 VLM 权重（训练入口加载检查点时会一并载入），P 用的是原版 Qwen3-VL（`apt_va` 训练时 VLM 冻结，没有 `vlm_weights`）。过拟合检查脚本保证评测时各自使用训练时的同一份 VLM。
4. `train.py` 中 `load_from_va` 之后的层映射打印保留（只在走扩展流程时出现）。

