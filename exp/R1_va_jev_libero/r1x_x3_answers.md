# R1-X / X3：代码核对的书面回答（工程侧，2026-11-12）

对应规格第 9.2 节 X3。依据为提交 `ea00265` 的代码，只读代码、未运行。结论：四项核对中**没有发现 bug**；有两处需要研究侧知道的事实，分别写在第 2、4 条末尾。

## 1. D 臂查询 VLA 的采样路径、K 与参考相机

- 路径：`examples/libero_r1/eval_r1.py::run_episode_vla` → `Policy.sample` → 服务端 `TrajPlanner.sample_trajs(share_vlm=True)`，即共享骨干路径。
- K = 1，每次查询执行前 3 步后重新查询。K = 1 时共享骨干与复制观测在数学上等价：骨干都只前向一次，复制 1 份即原张量。
- 参考相机：每次查询由客户端从 episode 种子派生的 VLA 参考相机随机流（`SeedSequence(seed).spawn(4)[3]`）均匀抽取 `agentview` 或 `eye_in_hand`，显式传给服务。
- 初始噪声来自 VLA 噪声流（`spawn(4)[1]`）。
- 已接受的 D 臂结果是在这套代码（提交 `2e57567` 之后）上跑的：服务器 10:59 更新到该提交，D 臂冒烟 11:05 开始。

## 2. 参考相机：训练时的规则，推理时为什么变化

- **训练**（`data_utils/dataset_base.py`，`sample_hdf5`，“read reference camera frame” 一段）：每个训练样本、每个末端执行器各自用 Python `random.sample` 从数据配置的 `ee_ref_cams` 中均匀抽 1 个相机。
  - Libero 的候选是 `(agentview, eye_in_hand)`。
  - 抽到的相机按名字在当前（可能已打乱顺序的）相机列表中定位；相机顺序由 `shuffle_cameras=True` 独立打乱，不影响抽取结果。
  - 预训练 `apt_va` 用的是各预训练数据集自己的候选列表，例如 DROID 是两个外部相机加腕部相机，规则相同。
- **推理**：动作以参考相机坐标系表达（`apt/action_transform.py::space_ee2cam`），推理也必须选一个参考相机。
  - 仓库原有的 `get_action` 走同一个 `DataSampler`，每次调用用服务端未设种子的 `random` 抽取，也就是沿用训练的均匀规则。
  - 提交 `2e57567` 起，R1 客户端改为从 episode 种子派生的独立随机流抽取：每次查询抽 1 次，32 条样本共用。分布仍是两相机各 1/2，只是变得可复现。
  - C 臂里 VA 查询和 VLA 参照各用一条流，各抽各的。
- **需要研究侧知道的事实**：同一噪声在两个参考相机下给出的轨迹差别很大（40k VA 中位数 247 mm，VLA 22 mm，见 2e57567 复验的检查 [4]）。这是动作坐标系不同的直接结果，不是 bug。

## 3. 客户端初始噪声与模型内部默认方式是否一致

- 客户端（`policy_client.Policy.noise`）：`numpy.random.Generator.standard_normal((K, nee, ta, act_dim))`，转 float32。
  - LIBERO 下形状为 (K, 1, 32, 10)；维度顺序为（样本, 末端执行器, 时间步, 动作维）；均值 0、标准差 1。
- 模型默认（`apt/action_expert.py::iterative_denoise`）：`torch.randn(B, Nee, Ta, act_dim)`。形状、维度顺序和分布都相同，只是随机数发生器不同（numpy PCG64 与 torch）。
- 服务端会断言噪声形状等于 `(K, Nee, Ta, act_dim)`，Nee 和 Ta 取自当次数据。
- 噪声原样作为 DDIM 链的起点。调度器：100 个训练步、20 个推理步、eta = 0、epsilon 预测、`clip_sample=False`；DDIM 的初始噪声尺度为 1，不需要额外缩放。
- 10 维动作的含义：参考相机坐标轴下、相对当前末端位姿的位移（米，3 维）+ 6D 旋转 + 夹爪（开合度 × 2 − 1）。

## 4. 转换数据每帧的对应关系与过滤空动作后的帧间隔

- 默认做法（`examples/libero_r1/convert_libero.py::replay_actions`，规格第 7 节裁决 1）：
  1. 先 `env.reset()`、`env.set_init_state(states[0])`。
  2. 对每个原始动作：若为空动作（OpenVLA 规则），**既不记录也不执行**，直接跳过。
  3. 否则先用当前观测 `obs` 生成一帧，再 `env.step(action)` 得到下一个 `obs`。
- **同一帧的图像、末端位姿、夹爪、相机内外参全部来自同一个 `obs` 和同一时刻的 `env.sim`**，由 `obs_libero2apt` 生成，与评测端喂给策略的打包函数相同。
- 时间戳写的是“已保留帧的序号” `0, 1, 2, …`，不是原始示教的步号。被跳过的空动作没有执行，仿真没有前进，所以相邻两帧之间恰好是一次实际执行的环境步（20 Hz）。
- 训练加载时 `record_dt` 由时间戳推得 1，`sample_dt=1.0`，所以 32 个未来状态 = 之后 32 次实际执行的环境步，与评测端（每个环境步一个点）一致。
- 本地用合成数据核对过：写出的 HDF5 经仓库自己的 `DataSampler.sample_hdf5` 读回后，未来状态与原始逐帧真值的最大误差为 2.4e-7。
- **一个推测（未核实，不影响结论）**：图像、末端位置和相机位姿都读自 MuJoCo 的 `sim.data`。按我对 MuJoCo 的理解，这些量可能对应 `env.step` 内最后一个物理子步积分之前的运动学，比关节状态晚一个物理子步（约 2 ms）。即便如此，训练数据和评测观测的读取方式完全相同，不构成训练与推理的不一致。
