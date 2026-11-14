# 本轮 VA（`va_libero4_r1c`）与 APT 发布的 Stage-0 的差别

本说明放在结果包顶层（规格第 17.5 节）。从第 17 节起，第一阶段所有使用 VA 的结果（R1-D、C 臂、A 臂）都来自 `va_libero4_r1c`。此前的 `va_libero4_r1`、`va_libero4_r1b` 已停用，它们的结果在 `archive_actual_pose/` 中，不与本轮结果混放。

## 两处差别

1. **注意力层加入锚定 RoPE**（配置字段 `stage0_action_rope = anchored`，写在检查点目录的 JSON 中，推理服务据此建模型并在启动日志中打印）。
   - APT 发布的 Stage-0：每个注意力头的前一半维度用 PRoPE，所有未来动作 token 共用同一个参考相机位姿；后一半维度不加位置编码，动作 token 之间也没有步序嵌入。去噪网络因此对 32 个未来 token 置换等变，只能生成无序的路点（规格 15、17.1；Z5、Z6、过拟合 P/P2 验证）。
   - 本轮：在后一半维度加 RoPE，动作 token 的一维位置从最后一个视觉 token 之后开始编号，与指令文字无关。冒烟 11 验证语言不泄漏：token 对齐时 VA 输出逐位相同；fp32 下不等长指令的位置最大差 ≤ 1e-4 m。
   - 不改变参数：RoPE 没有可学习参数，`apt_va` 的权重可以原样严格加载，实际也是从 `apt_va/ckpt_latest.pt` 严格加载后微调的。
2. **训练数据的未来轨迹取控制器指令目标**（`ee_pose_desired` / `gripper_desired`）。
   - APT 发布的 LIBERO 微调 VLA（`apt_vla_ftlibero`）实际学的是 OSC 控制器每一步的指令目标位姿（规格 12，R1-Y Y1：对指令目标第 1 步误差 0.74 cm，对实际位姿恒定偏 2.75 cm）。
   - 本轮 VA 的训练数据按同一口径：用官方示教动作重放（过滤空动作、丢弃重放失败的示教，共保留 1702 / 2000 个 episode），每一帧写入控制器在该步实际设定的目标（位置为夹爪 site 位置，姿态换算到 `robot0_eef_quat` 的 body 口径），夹爪目标为 (1 − 动作第 7 维) / 2。
   - 历史状态和观测仍是实际位姿与渲染图像。

## 其余设置

与规格 2.1 节相同：Stage-0、`base`、VLM 冻结、四个 suite 等权、`sample_multiplex=1000`、DDP 2 卡 × bs 32、学习率 1e-4、权重衰减 1e-2、预热 2000、40000 步、每 10000 步保存、fp16、不用 EMA、种子 0。所用检查点由规格 17.3 的训练后检查选出，见 `R1T3/check/r1c_selection.json`。
