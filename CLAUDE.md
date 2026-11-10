# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for APT (Action Expert Pretraining), a Vision-Language-Action policy: a Qwen3-VL backbone (`Qwen/Qwen3-VL-2B-Instruct`, pulled from Hugging Face at model construction) plus a diffusion action expert trained in two stages. There is no package install step, test suite, linter config, or CI — everything runs as modules from the repo root (`python -m apt.train`, `python -m apt.infer.remote_service`), and top-level packages (`data_utils`, `train_utils`, `infer_utils`, `shm_transport`) are imported by absolute name, so the working directory must be the repo root.

Training and inference require CUDA (`Trainer` hard-codes `cuda:<local_rank>`); nothing model-related runs on a Mac/CPU box.

## Commands

Environment (Python 3.10, see README for the torch/CUDA pin):

```bash
pip install -r requirements.txt
```

`deepspeed` and `xformers` are optional — `--backend ddp` works without DeepSpeed.

Two-stage pretraining / fine-tuning (wrappers around `apt.train`, launching via `torchrun` or the `deepspeed` launcher):

```bash
bash scripts/train.sh --backend ddp --gpus 0,1,2,3 --stage both --config pretrain --va-name apt_va --vla-name apt_vla
bash scripts/finetune.sh --backend deepspeed --gpus 0,1,2,3 --stage 1 --config finetune_pp --pretrained-ckpt apt_vla --vla-name ft_apt_pp --ds-zero 3 --vlm-mode lora --gc
```

The shell wrappers reject unknown flags. To override any other `TrainConfig` field (`--ckpt_dir`, `--log_dir`, `--dataloader_timeout`, `--workers`, …) call the trainer directly; fields are exposed through tyro:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 -m apt.train --backend ddp --config debug -s my_exp --train_stage 0 --bs 2
python -m apt.train --config debug --help   # argparse flags + all TrainConfig fields
```

Offline checkpoint sanity check (CPU only, no HF download — the closest thing to a test in this repo):

```bash
python scripts/test_ckpt.py --ckpt /path/ckpt_latest.pt --train-stage 1 [--load-from-va]
```

Inference server (needs a Pyro4 naming server first):

```bash
pyro4-ns -p 9091
python -m apt.infer.remote_service --ckpt checkpoints/APT/<name>/ckpt_latest.pt --uri control --ns_port 9091 --ensemble 4
```

Benchmarks run in *separate* conda envs and talk to that server through `shm_transport`: `bash examples/libero/test_libero.sh --benchmark libero|libero-pro|libero-plus ...` (from repo root) and `cd examples/PickPlace && bash eval_all.sh` (Isaac Sim's bundled Python). See the READMEs in those folders.

## Architecture

### Stage numbering (easy to get wrong)

The paper's "Stage 1 / Stage 2" are `train_stage=0` / `train_stage=1` in code, and `--stage 0|1|both` in the scripts.

- `train_stage=0` (VA prior): `HybridAttentionLayers` builds only N/2 attention blocks and language tokens are masked.
- `train_stage=1` (VLA likelihood): N interleaved blocks — even indices are newly inserted language-injection layers (RoPE), odd indices are the Stage-0 layers (PRoPE).
- `VLA.load_from_pretrain(..., load_from_va=True)` maps Stage-0 layer `i` → Stage-1 layer `2*i+1` under `dp_head.traj_context_attn.layers`. Only those attention layers are copied from the VA checkpoint; even-index layers and every other actor parameter keep the current model's initialization (the block marked `!!! IMPORTANT !!!` in `apt/vla.py`). Don't change this without asking.
- The `gate` parameter always has N rows; in stage 0, layer `i` uses gate row `2*i+1` so indices stay aligned across stages.
- `scripts/test_ckpt.py` hard-codes 5 / 10 layers (`vla_base`: VLM hidden states are tapped every 3rd LLM layer in `apt/vlm.py`, which sets N).

### Model

`apt/vla.py` `VLA` = `VLM` (`apt/vlm.py` → `apt/encoders/qwen3_vl.py`, modes `frozen` / `lora` / `full`) + `ActionExpert` (`apt/action_expert.py`). The VLM returns per-layer hidden states; each action-expert attention layer blends its vision-language stream with the matching VLM "highway" feature through a learnable sigmoid gate. The diffusion head is DDIM, epsilon-prediction, over a 10-dim per-end-effector action (3 translation + 6D rotation + gripper); `apt/action_transform.py` converts between that and the dataset's 17-dim EE state (flattened 4×4 world pose + gripper). Model sizes are the `vla_tiny|small|base` factories, selected by `TrainConfig.model`.

### Trainer

`apt/train.py` `Trainer` is backend-agnostic: all DDP-vs-DeepSpeed differences live in `_Backend` subclasses in `train_utils/backend.py` (distributed init, engine wrapping, micro-step/accumulation, state-dict gathering under ZeRO-3, optimizer save/restore). Keep it that way — no `if backend == ...` in `Trainer`.

CLI parsing is two-layer: argparse takes `--backend`, `--config`, `-s` (save name), `-c` (resume name), `--ds-config`; the remainder goes to `tyro` to override fields of the chosen `CONFIGS[...]` entry in `apt/configs.py`.

Checkpoint semantics:

- Written to `<ckpt_dir>/<name>/ckpt_latest.pt` (+ `ckpt_best.pt`, periodic `ckpt_<iter>.pt`); default root `./checkpoints/APT`, logs in `./logs/APT`. With no `-s`/`-c`, nothing is saved.
- `ckpt["weights"]` is the **actor only**. `ckpt["vlm_weights"]` exists only when the VLM was trained (LoRA adapters or full weights). DDP checkpoints embed optimizer/scheduler/scaler; DeepSpeed writes per-rank `*_opt_rank{N}.pt` shards alongside. Both formats load under either backend.
- `-c` resumes (iteration counter, optimizer, never re-expands VA layers); `--pretrained_ckpt` only loads weights and starts at iteration 0. Both accept a run name or a path ending in `.pt`.
- On first save the run's `TrainConfig` is dumped as `<launch-time>.json` next to the checkpoint. Inference depends on it: `apt/infer/planner.py` `parse_config` reads the newest `*.json` in the checkpoint directory to pick the model size, VLM mode, and the `DataConfig` of the *first* dataset class in the config. A `.pt` file moved without its JSON cannot be loaded by `TrajPlanner`.
- EMA is silently disabled under ZeRO-3.

### Data

- Per-episode HDF5 trajectories. `data_utils/dataset_base.py` holds `DataConfig`, `H5DatasetMapBase`, the sampler, and image processing; concrete datasets in `data_utils/datasets.py` each carry a class-level `config = DataConfig(...)` (cameras, EE indices, image size) and an `inst(train_stage)` classmethod. Some `inst` implementations select different file subsets per stage (e.g. `PickPlaceCan` uses the first half of files for stage 0).
- Dataset paths are resolved at import time in `data_utils/data_loc.py` from YAML keyed by host IP: `$APT_DATA_LOC` → `data_utils/data_loc.local.yaml` (gitignored; put personal paths here) → `data_utils/data_loc.yaml` (checked-in template with `YOUR_DIRECTORY` placeholders). Host detection picks the first `172.*` address; set `APT_DATA_LOC_HOST` to override. A miss only prints a warning — the failure comes later when a dataset is instantiated. (The main README still says to edit `data_loc.py`; the YAML is the current mechanism.)
- Adding a dataset: add a `DSET` enum member in `data_loc.py`, a YAML path entry, a `H5DatasetMapBase` subclass in `datasets.py`, and reference it from a `CONFIGS` entry. Config entries may name dataset classes as strings (that's how the dumped JSON round-trips).
- Mixed-dataset sampling uses `dataset_weights` with weighted/multiplex distributed samplers (`data_utils/dist_sampler.py`); `sample_multiplex` > 1 requires `dataset_weights`.

### Inference path

`TrajPlanner` (`apt/infer/planner.py`) always builds the model with `train_stage=1`, merges LoRA weights if present, buffers observation frames, and temporally ensembles predicted trajectories (`infer_utils/ensemble.py`). Loading a checkpoint also writes a gate visualization to `apt/gate.png` (untracked, not gitignored — don't commit it). `apt/infer/remote_service.py` exposes the planner over `shm_transport` (Pyro4 for RPC + shared memory for array payloads); evaluators in `examples/` are clients obtained via `shm_transport.get_shm_proxy` and run in their own simulator environments with only `shm_transport` on `PYTHONPATH`, so keep `shm_transport` free of torch/APT imports.
