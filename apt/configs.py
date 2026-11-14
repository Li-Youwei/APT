import os
import json
from typing import List, Dict
from dataclasses import dataclass, field, asdict, replace

from data_utils import datasets
from data_utils.dataset_base import H5DatasetMapBase


@dataclass
class TrainConfig(object):
    load_from_va: bool = False
    train_stage: int = 0  # 0: va pretrain, 1: vla finetune

    model: str = "base"  # choices are ["tiny", "small", "base"]
    pretrained_ckpt: str | None = None  # ckpt path of pretrained model

    # VLM (Qwen3-VL) fine-tuning mode:
    #   "frozen" - VLM weights are frozen, used only as a feature extractor (default)
    #   "lora"   - LoRA adapters are inserted into attention/FFN layers; only adapter weights are trained
    #   "full"   - All VLM weights are trained (full fine-tuning); requires more GPU memory
    vlm_finetune_mode: str = "frozen"
    use_gradient_checkpointing: bool = False  # save ~60% activation memory, ~30% slower
    vlm_lr: float | None = None  # separate LR for VLM params; None = same as max_lr
    vlm_wd: float = 1e-10  # weight decay for VLM params; near-zero following openpi convention

    bs: int = 64  # batch size per gpu and per fwd
    workers: int = 8  # num_workers
    persistent_workers: bool = True  # CRITICAL: Set to False to prevent worker memory accumulation
    pin_memory: bool = True
    prefetch_factor: int = 4  # Reduced from 4 to save memory
    dataloader_timeout: int = 300  # DataLoader worker timeout in seconds (only used when workers > 0)
    bind_cpu_affinity: bool = True
    fp16: bool = True  # enable mixed precision training (fp32 and bfloat16)
    use_prefetcher: bool = True  # CUDA-stream prefetcher to overlap CPU→GPU transfer with compute
    use_compile: bool = False  # torch.compile the action expert (~15-25% speedup, longer warmup)

    grad_clip: float = 1.0  # <= 0 disables the grad clip
    max_lr: float = 1e-4  # maximum learning rate
    wd: float = 1e-2  # weight decay
    num_warmup: int = int(10e3)  # warm up steps
    gradient_accumulation_steps: int = 1

    ema_enabled: bool = False
    ema_start: int = int(400e3)
    ema_decay: float = 0.9995

    dataset_classes: List[type[H5DatasetMapBase] | str] = field(default_factory=list)
    dataset_weights: List[float] | None = None  # len = len(datasets)
    sample_multiplex: int = 1   # set this to a large number (e.g. 1000) if the total number of samples are small

    log_interval: int = 100
    save_interval: int = int(100e3)  # ckpt are named as ckpt_{iter}.pt, set <0 to disable this
    save_latest_interval: int = 1000  # ckpt are named as ckpt_latest.pt
    max_iterations: int = int(600e3)
    log_dir: str = "./logs/APT"        # TensorBoard log root; per-experiment subdir is appended
    ckpt_dir: str = "./checkpoints/APT"  # Checkpoint root; per-experiment subdir is appended

    robot_pose_aug: bool = False  # if True, add noise to the robot states
    camera_view_dropout: float = 0.0  # probability of dropping one camera view during training

    seed: int | None = None  # None = unseeded (old behaviour); else torch/numpy/random seeded with seed + rank
    # Stage-0 only: RoPE on the free half of the attention head dims for the action tokens
    # (exp/R1_va_jev_libero/spec.md 15.3). "off" = original; "context" = positions as the context
    # encoder computes them (offset after the last VL token, i.e. after the instruction text);
    # "anchored" = positions start after the last vision token (instruction-length independent).
    stage0_action_rope: str = "off"
    
    def __post_init__(self):
        for i, D in enumerate(self.dataset_classes):
            if isinstance(D, str):
                self.dataset_classes[i] = getattr(datasets, D)
            else:
                assert issubclass(D, H5DatasetMapBase)
    
    def dump(self, path: str):
        items = asdict(self)
        dataset_classes = items["dataset_classes"]
        for i, D in enumerate(dataset_classes):
            if issubclass(D, H5DatasetMapBase):
                dataset_classes[i] = D.__name__
            else:
                assert isinstance(D, str)
        
        save_folder = os.path.dirname(path)
        os.makedirs(save_folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(items, fp, ensure_ascii=False, indent=4)
    
    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as fp:
            items = json.load(fp)
        return cls(**items)


CONFIGS: Dict[str, TrainConfig] = {}
CONFIGS["debug"] = TrainConfig()
CONFIGS["pretrain"] = TrainConfig(
    dataset_classes=[
        datasets.Droid,
        datasets.AgibotWorldAlpha,
        datasets.InternA1Franka,
        datasets.InternA1Lift2,
        datasets.InternA1Genie1,
        datasets.InternA1SplitAloha,
        datasets.InternM1Franka,
    ],
    dataset_weights=[10, 10, 2, 2, 2, 2, 2]
)
CONFIGS["finetune_pp"] = TrainConfig(
    dataset_classes=[
        datasets.PickPlaceCan,
    ],
    dataset_weights=[1]
)
CONFIGS["finetune_libero"] = TrainConfig(
    dataset_classes=[datasets.LiberoSpatial, datasets.LiberoObject, datasets.LiberoGoal, datasets.Libero10],
    dataset_weights=[1, 1, 1, 1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
)
CONFIGS["finetune_libero_intern"] = TrainConfig(
    dataset_classes=[datasets.LiberoSpatial, datasets.LiberoObject, datasets.LiberoGoal, datasets.Libero10, datasets.InternA1Franka],
    dataset_weights=[1, 1, 1, 1, 1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
)
CONFIGS["finetune_libero_spatial"] = TrainConfig(
    dataset_classes=[datasets.LiberoSpatial],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
)
CONFIGS["finetune_libero_object"] = TrainConfig(
    dataset_classes=[datasets.LiberoObject],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
    camera_view_dropout=0.15,  # per-sample prob from dataset takes priority in mixed training
)
CONFIGS["finetune_libero_goal"] = TrainConfig(
    dataset_classes=[datasets.LiberoGoal],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
)
CONFIGS["finetune_libero_10"] = TrainConfig(
    dataset_classes=[datasets.Libero10],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
)
CONFIGS["finetune_aloha_pp"] = TrainConfig(
    dataset_classes=[
        datasets.AlohaPickPlace,          # dual arm
    ],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
)
CONFIGS["finetune_aloha_pp_storage"] = TrainConfig(
    dataset_classes=[
        datasets.AlohaPickPlace,          # dual arm
        datasets.AlohaTableStorage,       # initial
    ],
    dataset_weights=[1, 1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
)
CONFIGS["finetune_aloha_pp_clutter"] = TrainConfig(
    dataset_classes=[
        datasets.AlohaPickPlaceClutter,          # dual arm
    ],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
)

# R1 (exp/R1_va_jev_libero/spec.md, section 2.1): Stage-0 VA fine-tuned on the
# four LIBERO suites, initialised from HF `KechunXu1/apt_models/apt_va` via
# --pretrained_ckpt (no layer mapping). 2 GPUs x bs 32 under DDP.
CONFIGS["r1_va_libero4"] = TrainConfig(
    train_stage=0,
    model="base",
    vlm_finetune_mode="frozen",
    dataset_classes=[datasets.LiberoSpatial, datasets.LiberoObject, datasets.LiberoGoal, datasets.Libero10],
    dataset_weights=[1, 1, 1, 1],
    sample_multiplex=1000,
    bs=32,
    gradient_accumulation_steps=1,
    max_lr=1e-4,
    wd=1e-2,
    num_warmup=int(2e3),
    max_iterations=int(40e3),
    save_interval=int(10e3),
    fp16=True,
    ema_enabled=False,
    camera_view_dropout=0.0,
    robot_pose_aug=False,
    seed=0,
)

# R1b (spec section 12.3): identical to r1_va_libero4 except 60k iterations; trained on the
# re-converted data that carries ee_pose_desired / gripper_desired (point APT_DATA_LOC at it).
# Save name: va_libero4_r1b.
CONFIGS["r1b_va_libero4"] = replace(CONFIGS["r1_va_libero4"], max_iterations=int(60e3))

# R1-Z overfit test (spec sections 13.3 / 14): LIBERO-Goal task 1 only, 1 GPU, bs 64, 8k steps.
# P = Stage-0 from apt_va loaded directly; Q = Stage-1 from the pretrained VLA apt_vla loaded
# directly (no load_from_va). Pass --pretrained_ckpt <apt_va | apt_vla> on the command line.
CONFIGS["overfit_s0_goal1"] = replace(
    CONFIGS["r1_va_libero4"], dataset_classes=[datasets.LiberoGoalTask1], dataset_weights=[1],
    bs=64, max_iterations=8000, num_warmup=500, save_interval=2000)
CONFIGS["overfit_s1_goal1"] = replace(CONFIGS["overfit_s0_goal1"], train_stage=1, load_from_va=False)

# R1-Z P2 (spec sections 15.3 / 16): P with Stage-0 action RoPE, "anchored" mode only. The
# "context" mode leaks the instruction's token count into the VA and must not be used for any
# VA training or evaluation (spec 16.2); no config entry is provided for it.
CONFIGS["overfit_s0ropeA_goal1"] = replace(CONFIGS["overfit_s0_goal1"], stage0_action_rope="anchored")

# R1-T3 (spec section 17.2): the R1 VA recipe (section 2.1: 4 LIBERO suites, 2 GPUs x bs 32, 40k
# steps, save every 10k, seed 0) with anchored Stage-0 action RoPE, trained on the commanded-target
# data (point APT_DATA_LOC at it) from apt_va. Save name: va_libero4_r1c.
CONFIGS["r1c_va_libero4"] = replace(CONFIGS["r1_va_libero4"], stage0_action_rope="anchored")

