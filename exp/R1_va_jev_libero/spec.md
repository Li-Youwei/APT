# 【实验规格】R1：VA + Jev 在 LIBERO 上的首轮闭环能力测试

本规格交给工程侧实现。所有数值都是定值，不需要工程侧自行选择。代码推送到 GitHub 后由 Codex 在服务器（2×4090，只装了原版 LIBERO 环境）上执行。

**分两个阶段。第一阶段只实现和执行 R1-T、R1-D 以及 R1-C 的 A、C、D 三个臂。B 臂（VA + Jev）属于第二阶段：本次不实现、不执行，等研究侧看完第一阶段结果包后另行通知。** 文中标有“（第二阶段）”的内容本次全部跳过。

## 1. 实验编号与目标

R1 由三个子实验组成，按顺序执行。

| 编号 | 内容 | 要回答的问题 |
| --- | --- | --- |
| R1-T | 重新微调 VA（4 个 suite 联合） | 得到后续所有实验用的 VA 权重 |
| R1-D | VA 多模性离线诊断（开环，不调 Jev） | 新 VA 是否多模；正确行为是否在样本里 |
| R1-C | 闭环评测，第一阶段 3 个臂，第二阶段加 1 个臂 | 第一阶段：VA 候选在完美选择下离微调 VLA 有多远；第二阶段：VA + Jev 的成功率 |

R1-C 的四个方法臂：

| 臂 | 候选来源 | 选择器 | 作用 |
| --- | --- | --- | --- |
| A `va_random` | VA | 均匀随机取一条 | 下界：没有选择器时 VA 能做到多少 |
| B `va_jev`（第二阶段） | VA | Jev | 待测方法 |
| C `va_oracle` | VA | 取离微调 VLA 预测最近的候选 | 上界：选择器完美时 VA 候选能做到多少 |
| D `vla` | 微调 VLA `apt_vla_ftlibero` | 无 | 参照系，同时检查环境与协议是否正确 |

**预期结果**

- D：四个 suite 平均 ≥ 93%（论文为 96.1%）。
- A：Goal ≤ 40%；Spatial、Object、Long 可能明显高于 Goal。
- C：每个 suite 不低于 D 减 15 个百分点。
- B（第二阶段）：每个 suite 都满足 A ≤ B ≤ C。

**不符合预期的情况（出现任意一条即在结果包中标红，不要自行修改设计）**

1. D 平均 < 90%：怀疑环境、协议或动作转换有 bug。
2. A 在 Goal 上 > 60%：怀疑 VA 不是多模的，或语言信息泄漏进了 VA。
3. C 在任一 suite 上 < 70%：怀疑 VA 能力不足、聚类阈值不合适或候选提前被截断。
4. （第二阶段）B < A（差值超过 5 个百分点）：怀疑叠加图投影、选项映射或 Jev 调用有 bug。
5. R1-D 中 Goal 的初始帧平均聚类数 < 2。

## 2. 实验配置

### 2.1 R1-T：VA 微调

| 项 | 值 |
| --- | --- |
| 初始化权重 | Hugging Face `KechunXu1/apt_models` 的 `apt_va/ckpt_latest.pt`（预训练 VA，Stage-0），作为 `pretrained_ckpt` 直接加载，从第 0 步开始训练 |
| `train_stage` | 0 |
| `model` | `base` |
| `vlm_finetune_mode` | `frozen` |
| 数据集 | `LiberoSpatial`、`LiberoObject`、`LiberoGoal`、`Libero10`，`dataset_weights=[1,1,1,1]`，`sample_multiplex=1000` |
| 后端与 GPU | `ddp`，2 张 GPU |
| `bs` | 每张 GPU 32，`gradient_accumulation_steps=1` |
| 显存不足时 | 改为每张 GPU 16、`gradient_accumulation_steps=2`，其余不变，并在结果包中注明 |
| `max_lr` / `wd` / `num_warmup` | 1e-4 / 1e-2 / 2000 |
| `max_iterations` | 40000 |
| `save_interval` | 10000（得到 10k、20k、30k、40k 四个检查点） |
| `fp16` | True |
| `ema_enabled` | False |
| `camera_view_dropout` / `robot_pose_aug` | 0.0 / False |
| 保存名 | `va_libero4_r1` |
| 随机种子 | 0 |

R1-C 使用 40k 检查点。R1-D 使用全部四个检查点。

### 2.2 R1-D：多模性离线诊断

| 项 | 值 |
| --- | --- |
| suite | `libero_spatial`、`libero_object`、`libero_goal`、`libero_10` |
| 状态 | 每个任务的前 5 个固定初始状态，重置后先执行 10 步空动作，取此时的观测 |
| VA 检查点 | 10k、20k、30k、40k |
| 每个状态的采样数 | 64 条 |
| 聚类 | 2.4 节的方法；另外用阈值 0.04 m 和 0.12 m 各重算一次聚类数 |
| 参照轨迹 | 微调 VLA 在同一观测、真实指令下预测的 1 条轨迹 |
| 覆盖判定 | 64 条中存在一条与参照轨迹的距离（2.4 节定义）≤ 0.08 m，记为已覆盖 |
| 采样随机种子 | `10000 × suite序号 + 100 × task_id + state_id`（suite 序号按上面顺序 0–3） |

总量：4 suite × 10 任务 × 5 状态 × 4 检查点 = 800 次采样。

### 2.3 R1-C：闭环评测协议

| 项 | 值 |
| --- | --- |
| suite | `libero_spatial`、`libero_object`、`libero_goal`、`libero_10` |
| 任务与 episode | 每个 suite 10 个任务，每个任务使用全部 50 个固定初始状态（下标 0–49） |
| 总量 | 每臂 2000 个 episode；第一阶段三臂共 6000 个，第二阶段再加 2000 个 |
| 重置后等待 | 10 步空动作，不计入步数 |
| 最大步数 | Spatial 220、Object 280、Goal 300、Long 520 |
| 成功判定 | 环境返回 `done=True` |
| 图像分辨率 | 策略输入 256×256（`agentview` 与 `robot0_eye_in_hand`） |
| 种子数 | 1 个 |
| 每个 episode 的采样随机种子 | `100000 × suite序号 + 1000 × task_id + episode_id` |
| VA 检查点 | `va_libero4_r1` 的 40k |
| VLA 检查点 | Hugging Face `KechunXu1/apt_models` 的 `apt_vla_ftlibero/ckpt_latest.pt` |
| 扩散推理步数 | 20（模型默认值） |
| 轨迹时间集成 | 关闭 |

**D 臂**：沿用仓库现有评测逻辑，每次查询采 1 条，执行前 3 步后重新查询。只把最大步数改为上表的值。

**A、B、C 臂**共用同一套流程，只有“选择器”不同：

| 项 | 值 |
| --- | --- |
| 重规划间隔 | 每 8 个环境步查询一次 VA |
| 每次查询采样数 K | 32 |
| 每条轨迹长度 | 32 步 |
| 执行 | 执行被选中轨迹的前 8 步，然后重新查询 |
| 候选生成（B、C 臂） | 对 32 条样本按 2.4 节聚类，每个簇取中心样本；按簇大小降序最多保留 6 个候选，大小相同按簇内最小样本下标升序 |
| A 臂的选择 | 直接取 32 条样本中下标为 0 的那条，不聚类 |
| B 臂的选择（第二阶段） | 候选数为 1 时不调用 Jev，直接执行；候选数 ≥ 2 时按 2.5 节调用 Jev |
| C 臂的选择 | 用同一观测、真实指令查询 VLA 得到 1 条参照轨迹，选与它距离（2.4 节定义）最小的候选；候选数为 1 时也照常查询并记录距离 |

### 2.4 轨迹距离与聚类

- 轨迹位置取世界坐标系下末端执行器位置，单位米。
- 夹爪二值化：开合度 > 0.8 记为张开，否则记为闭合。
- 两条轨迹的距离：32 步中逐步位置欧氏距离的最大值。若两条轨迹夹爪二值状态不同的步数 > 8，则距离直接记为 1.0 m。
- 聚类：完全连接（complete linkage）的层次聚类，距离阈值 0.08 m。
- 簇中心样本：簇内到其他成员距离之和最小的那条；并列时取下标最小的。

### 2.5 Jev 调用（B 臂，第二阶段）

| 项 | 值 |
| --- | --- |
| 模型 | `jev-1.14.0`（固定版本号，不用别名） |
| 问题类型 | Choice |
| 选项 | `opt_1` … `opt_N`（N 为候选数，2–6）加 `none` |
| 每次决策的调用数 | 3 次，选项顺序各不相同，并发发出 |
| 选项顺序的随机种子 | `episode 种子 × 1000 + 决策序号`，三次调用分别用该种子生成的前三个排列 |
| 最终选择 | 把三次调用的概率按候选对齐后取平均，在 `opt_1`…`opt_N` 中取平均概率最大的；`none` 的概率只记录，不参与取最大 |
| 并列 | 取簇更大的候选 |
| 失败重试 | 单次调用最多重试 3 次，间隔 2 s、4 s、8 s |
| 重试后仍失败 | 用剩余成功调用的结果；三次全失败则执行最大簇的候选，并记录 `jev_fallback=true` |
| 深度图 | 本轮不使用 |

**每次调用的输入内容**（顺序固定）：

1. 指令文字：环境给出的任务描述原文。
2. 1 张干净的当前 `agentview` 图。
3. N 张叠加图，每张只画一条候选，与选项一一对应。
4. 图例与问题文字，原文如下：

> The robot is partway through carrying out the instruction. Each option image shows one candidate path for the robot gripper over the next 1.6 seconds, drawn on the current camera view. The magenta line with arrowheads is the path of the gripper; it starts at the small magenta dot and ends at the magenta cross. A hollow circle marks where the gripper closes to grasp, and a hollow square marks where the gripper opens to release. Which option's path makes correct progress toward completing the instruction? Choose none only if no path makes progress.

`none` 选项的文字：`None of the drawn paths makes progress on the instruction.`

选项名和图片文件名不得包含簇大小、样本下标或任何可能泄露答案的信息。

**叠加图画法**

| 项 | 值 |
| --- | --- |
| 底图 | 当前 `agentview` 的 256×256 图，双三次插值放大到 512×512，PNG |
| 投影 | 用 `agentview` 的内参和外参把 32 个末端位置投影到图上，坐标随放大乘 2 |
| 路径 | 品红色 (255, 0, 255) 线宽 4 px，外加 1 px 白边 |
| 起点 | 半径 5 px 的品红实心圆，位置为当前末端位置 |
| 箭头 | 沿路径等弧长放 4 个 |
| 终点 | 臂长 8 px 的品红叉 |
| 抓取标记 | 夹爪二值状态由张开变闭合的那一步，半径 14 px 的空心圆，黑白双圈 |
| 释放标记 | 夹爪二值状态由闭合变张开的那一步，边长 28 px 的空心方框，黑白双圈 |
| 出界点 | 投影落在图外的点裁剪到图像边界 |

## 3. 需要实现或修改的功能

只描述行为与接口，写法由工程侧决定。不得修改 `apt/vla.py` 中标有 `!!! IMPORTANT !!!` 的现有加载逻辑，新功能另加入口。

### 3.1 加载预训练 VA

- 从 Hugging Face 下载 `apt_va` 整个目录（权重与同目录的 JSON 配置）。
- 用仓库现有的 `--pretrained_ckpt` 机制加载，`train_stage=0`，不做任何层映射。
- 加载必须是严格模式且成功。若该目录下的权重文件名不是 `ckpt_latest.pt`，使用目录中迭代数最大的 `.pt` 文件，并在结果包中注明文件名。
- 正式训练前先用 `scripts/test_ckpt.py --train-stage 0` 检查该权重。

### 3.2 LIBERO 训练数据

- 训练需要仓库格式的逐 episode HDF5，相机为 `agentview` 与 `eye_in_hand`，256×256，带相机外参。
- 如果服务器上没有这份数据，需要实现转换：在仿真中按官方示教的状态序列重放，重新渲染图像并记录末端位姿、夹爪开合度、相机内外参与语言指令。
- 转换后在 `data_utils/data_loc.local.yaml` 中登记四个 suite 的路径。
- 结果包中报告每个 suite 的 episode 数（预期各 500；若过滤了重放失败的示教，报告过滤数）。

### 3.3 推理端支持 VA

- 推理端根据检查点目录里 JSON 配置的 `train_stage` 建模型，使 Stage-0 检查点可以加载。Stage-1 检查点的行为保持不变。
- 一次查询可以返回 K 条轨迹，每条使用独立的初始噪声。
- 支持在查询前设置随机种子，使同一种子、同一观测下的 K 条样本可复现。
- 视觉语言骨干对同一观测只前向一次，K 条样本共用其输出。如果这项优化在 2 小时内做不完，退回到把观测复制 K 份的现有做法，并在结果包中注明。
- VA 与 VLA 两个服务可以同时运行（各占一张 GPU），评测端能同时连接两者。

### 3.4 评测端

- 一个命令行参数选择方法臂：`va_random`、`va_oracle`、`vla`；`va_jev` 在第二阶段加入，第一阶段的代码结构要为它留出选择器接口。
- 可指定 suite、任务下标范围、每任务 episode 数、输出目录。
- 按 2.3–2.4 节实现重规划、聚类、候选生成，以及 A、C 两种选择器。
- 支持断点续跑：已完成的 episode 不重复执行。
- （第二阶段）Jev 的密钥从环境变量读取，不得写入代码、日志或结果包。

### 3.5 离线诊断脚本

- 按 2.2 节遍历状态与检查点，输出 4.2 节的表格与图。

## 4. 需要记录的输出

所有输出放在 `results/R1/` 下，结果包即该目录的压缩包（不含权重）。

### 4.1 R1-T

- 训练配置 JSON、TensorBoard 日志。
- 每 1000 步的 `pos_loss`、`rot_loss`、`openness_loss`、`total_loss`（CSV）。
- 每 1 万步的墙钟时间、每张 GPU 的峰值显存。

### 4.2 R1-D

每个（suite, 任务, 状态, 检查点）一行的 CSV，字段：

- `suite`、`task_id`、`state_id`、`ckpt_iter`、`instruction`
- `n_clusters_004`、`n_clusters_008`、`n_clusters_012`
- `cluster_sizes_008`（降序列表）
- `covered`（0/1）、`min_dist_to_vla`（米）
- `vla_cluster_rank`（参照轨迹最近的簇按大小排第几，从 1 开始）、`vla_cluster_share`（该簇占 64 条的比例）

汇总表：每个 suite × 检查点的平均聚类数、覆盖率、`vla_cluster_share` 的均值与最小值。

可视化：每个状态一张 512×512 图，把 64 条样本画在 `agentview` 上，同簇同色，VLA 参照轨迹用白色粗线。只保存 40k 检查点的图（200 张）。

### 4.3 R1-C

**episode 级 CSV**（每臂每 suite 一个文件），字段：

- `suite`、`task_id`、`episode_id`、`instruction`、`arm`
- `success`（0/1）、`steps`、`n_queries`
- `n_decisions`（候选数 ≥ 2 的查询次数）；第二阶段另加 `n_jev_calls`、`n_jev_fallback`
- `wall_time_s`

**查询级 JSONL**（A、B、C 臂，每次查询一行），字段：

- `suite`、`task_id`、`episode_id`、`step`、`query_idx`
- `n_clusters`、`cluster_sizes`
- `candidates`：每个候选的簇大小、中心样本下标、32 步末端位置与夹爪开合度
- `selected_candidate`（按簇大小排序后的下标）
- `oracle_candidate` 与各候选到 VLA 参照轨迹的距离（C 臂记录；第二阶段 B 臂也记录，为此需要额外查询 VLA，但不得用它影响选择）
- `oracle_dist`（C 臂：被选候选到 VLA 参照轨迹的距离，米）
- `agree_with_oracle`（第二阶段 B 臂：Jev 选择是否等于 `oracle_candidate`）
- `dist_selected_to_prev`（本次选中轨迹的前 24 步与上次选中轨迹的第 9–32 步的最大位置距离，米；首次查询记空）
- `sampling_time_s`

**Jev 原始记录**（第二阶段 B 臂，每次调用一行 JSONL）：

- 完整请求内容（图片以文件路径引用）、选项顺序、完整原始返回、各选项概率、`none` 概率
- 延迟（秒）、输入 token 数、重试次数、响应中的 `model` 字段

**图片**：（第二阶段）B 臂每次决策保存干净图与全部叠加图。

**视频**：每臂每个任务保存前 3 个 episode 的视频，A、C 臂的视频上叠加当次选中的轨迹。

**汇总表**

1. 成功率：臂 × suite，附 95% Wilson 区间和四 suite 平均。
2. 成功率：臂 × 任务（每个 suite 一张）。
3. （第二阶段）B 臂：每个 suite 的每 episode 平均决策数、Jev 调用数、与 oracle 的一致率、`none` 平均概率、回退率、延迟 p50 与 p95、总 token 数。
4. A、C 臂：每个 suite 的平均聚类数随步数的变化（每 8 步一个点）。
5. C 臂：每个 suite 的 `oracle_dist` 分布（均值、p50、p95），以及成功与失败 episode 分开统计的均值。
6. C 臂相对 D 臂的逐任务成功率差值表（按差值从大到小排序）。
7. （第二阶段）B 臂失败 episode 的归因计数，规则如下：
   - 整个 episode 中 Jev 与 oracle 一致率 ≥ 90%：记为“执行或候选问题”；
   - 否则记为“选择问题”。

## 5. 冒烟测试标准

全部通过后才启动正式运行。任意一条不通过，把现象交给工程侧调试。

**R1-T**

1. `apt_va` 权重通过 `scripts/test_ckpt.py --train-stage 0` 检查，Stage-0 模型严格加载成功。
2. 训练 200 步无报错，`total_loss` 第 200 步的值低于第 1 步。

**R1-D**

3. 用 200 步的检查点在 `libero_goal` 任务 0 的状态 0 上跑通，生成一行 CSV 和一张可视化图。
4. 人工看图：轨迹起点落在图中夹爪位置上（投影正确的标志）。

**R1-C**

5. D 臂：`libero_goal` 任务 0 和任务 1 各 3 个 episode，成功数 ≥ 5/6。
6. C 臂（VA 用 40k 检查点）：`libero_goal` 任务 0 和任务 1 各 3 个 episode，全部跑完无报错，成功数 ≥ 3/6。
7. （第二阶段）B 臂：`libero_goal` 任务 0 和任务 1 各 2 个 episode，全部跑完；至少发生 1 次 Jev 调用；每次调用的原始返回、图片都已落盘；三次调用的概率在对齐选项后可以正确平均（抽查 1 次决策，手算核对）。
8. （第二阶段）B 臂日志中 `jev_fallback` 的比例 ≤ 10%。正式运行中若某个 suite 超过 2%，该 suite 结果作废并重跑。
9. 同一 episode 用相同种子重跑 A 臂两次，`steps` 与 `success` 完全一致。
10. 断点续跑：中途终止后重启，已完成的 episode 被跳过。

## 6. 执行顺序

**第一阶段**

1. R1-T 冒烟 → R1-T 正式训练。
2. D 臂冒烟 → D 臂全量（先确认参照系）。
3. R1-D 全量。
4. C 臂冒烟 → C 臂全量。
5. A 臂全量。
6. 打包 `results/R1/` 作为第一阶段结果包，交回研究侧。到此停止。

如果算力紧张需要排优先级：D 臂与 C 臂优先于 A 臂，`libero_goal` 优先于其他 suite。

**第二阶段（收到研究侧通知后才开始）**

7. 实现 B 臂 → B 臂冒烟 → B 臂全量 → 追加打包。

研究侧放行第二阶段的判据（供参考，不需要工程侧判断）：C 臂四个 suite 平均成功率不低于 D 臂减 15 个百分点。

## 7. 修订记录：对工程侧 8 条疑问的裁决（2026-11-10）

以下裁决优先于正文中与之冲突的表述。

1. **数据转换方式**：采用 OpenVLA 的做法，即动作重放、过滤空动作、丢弃重放后不成功的示教。理由：仓库数据路径模板是 `*_no_noops`，VA 必须与对照的 `apt_vla_ftlibero` 用同一口径的数据。3.2 节中“按状态序列重放”作废。结果包中报告每个 suite 保留与丢弃的 episode 数。
2. **VA 的指令输入**：查询 VA 时传真实指令，与训练一致。新增冒烟 11：同一观测、同一种子下，分别传真实指令和字符串 `do nothing`，VA 返回的 32 条轨迹逐元素最大差值 ≤ 1e-4 米。不通过说明语言泄漏进了 VA，停止并上报。
3. **种子用法**：同意。每个 episode 一个种子，派生两条独立随机数流，VA 与 VLA 各用一条，episode 内多次查询依次取噪声。D 臂同样处理。
4. **A 臂的 `selected_candidate`**：同意按所在簇的大小排名记录，并额外记录实际执行的轨迹。
5. **`vla_cluster_rank`**：同意，定义为离参照轨迹最近的那条样本所在簇的大小排名。
6. **两个边界**：聚类合并条件为距离 ≤ 阈值。最大步数用 `t >= 上限` 判定，即恰好执行到上限步数为止，A、C、D 三臂一致。
7. **训练 loss 口径**：冒烟 2 改为比较第 1–20 步的 `total_loss` 均值与第 181–200 步的均值，后者必须更低。正式 CSV 每 1000 步记一次、取前 100 步均值，同意。
8. **冒烟 9、10 的位置**：同意放在 C 臂冒烟之后、C 臂全量之前。冒烟 11 放在 R1-D 冒烟之后。
