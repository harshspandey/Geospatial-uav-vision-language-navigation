# Geographic Prior Injection for UAV Vision-Language Navigation

This repository contains a course project extension of FlightGPT for CityNav navigation. The main idea is simple: inject CityRefer landmark coordinates into the prompt so the model treats them as a spatial hint rather than as the final target location.

The implementation lives in `FlightGPT_GeoPrior/`. The top level also includes the final report and sample map figures.

## Overview

- Base system: FlightGPT
- Task: UAV vision-language navigation on CityNav
- Method: supervised fine-tuning (SFT) followed by GRPO
- Geographic prior source: `data/cityrefer/objects.json`
- Prompt behavior: inject landmark pixel coordinates before the image input

Example prior:

```text
[Geographic Prior] 'Leslie Road' is at map pixel [1054, 228].
The target building is located NEAR this landmark - search within ~400 pixels of this location.
```

The reward shaping in GRPO is designed so that predicting the landmark itself is not enough; the model must still localize the actual target.

## Results

Evaluation on 5311 test episodes:

| Split | SR | OSR | NE | SPL |
|---|---:|---:|---:|---:|
| Easy | 22.99% | 42.41% | 55.20 m | 20.28% |
| Medium | 19.93% | 35.47% | 69.25 m | 18.39% |
| Hard | 22.56% | 34.27% | 77.68 m | 21.30% |
| Overall | 21.90% | 37.86% | 66.14 m | 19.97% |

Baseline reference from the project notes: FlightGPT GRPO without geographic prior achieved `SR=21.20%` and `NE=76.20 m`.

## Repository Layout

```text
.
├── FlightGPT_GeoPrior/          # Main codebase for training, evaluation, and demo
├── README.md                    # This overview
├── VG_HPM_IEEE_Report_Final.pdf # Final report
├── VG_HPM_IEEE_Report_Final.tex # Report source
├── landmark_map_cropped.jpg     # Figure asset
└── plain_map_cropped.jpg        # Figure asset
```

Inside `FlightGPT_GeoPrior/`:

```text
FlightGPT_GeoPrior/
├── run_sft_geo.sh
├── merge_sft_geo.sh
├── run_grpo_geo.sh
├── eval_geo_trained.py
├── eval.py
├── live_demo.py
├── requirements.txt
├── data/
├── navgym/
├── open-r1-multimodal/
└── LLaMA-Factory/
```

## What Is Included

- Training and evaluation code for the geo-prior variant
- CityNav split metadata under `FlightGPT_GeoPrior/data/citynav/`
- Geo-prior RL data at `FlightGPT_GeoPrior/data/training_data/citynav_rl_data_geo.json`
- CityRefer object metadata at `FlightGPT_GeoPrior/data/cityrefer/objects.json`
- LLaMA-Factory configs for SFT and LoRA merge
- GRPO training entrypoint for the geo-prior setup

## What Is Not Fully Self-Contained

This repository is not yet fully plug-and-play on a fresh machine.

- Large artifacts such as `model_weight/`, `R1PhotoData/`, and generated `saves/` directories are expected by the scripts but are not checked into the repo.
- Several scripts and configs still contain hardcoded absolute paths from the original training server.
- Some instructions assume a pre-configured Conda environment and a single-GPU server with at least 24 GB VRAM.

If you want public reproducibility, this is the main gap to fix next.

## Environment

The checked-in environment file is `FlightGPT_GeoPrior/requirements.txt`. The current stack in the repo is centered around:

- Python 3.11
- `torch==2.6.0`
- `transformers==4.50.0`
- `trl==0.17.0`
- FlashAttention 2 wheel for CUDA 12 / Torch 2.6

Suggested setup:

```bash
cd FlightGPT_GeoPrior
conda create -n dlcv_vghpm python=3.11
conda activate dlcv_vghpm
pip install -r requirements.txt
pip install vllm==0.4.0
```

Then install the editable subpackages used by the training pipeline:

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ../open-r1-multimodal
pip install -e .
cd ..
```

## Required Assets

Before training or evaluation, you will need:

1. Base FlightGPT model weights in `FlightGPT_GeoPrior/model_weight/`
2. CityNav map imagery in `FlightGPT_GeoPrior/R1PhotoData/`
3. Writeable output directories such as `FlightGPT_GeoPrior/logs/` and `FlightGPT_GeoPrior/saves/`

The code also assumes the SFT dataset file:

- `FlightGPT_GeoPrior/LLaMA-Factory/data/vghpm_sft_v4_geo.json`

## Hardcoded Paths You Must Patch

On a new machine, update the absolute `/home/priyanka/.../FlightGPT_GeoPrior` paths in these files before running anything:

- `FlightGPT_GeoPrior/run_sft_geo.sh`
- `FlightGPT_GeoPrior/merge_sft_geo.sh`
- `FlightGPT_GeoPrior/run_grpo_geo.sh`
- `FlightGPT_GeoPrior/LLaMA-Factory/examples/train_lora/sft_geo.yaml`
- `FlightGPT_GeoPrior/LLaMA-Factory/examples/merge_lora/sft_geo_merge.yaml`
- `FlightGPT_GeoPrior/open-r1-multimodal/src/open_r1/grpo_jsonl_citynav_geo.py`

One simple approach:

```bash
cd FlightGPT_GeoPrior
MYPATH=$(pwd)

sed -i "s|/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior|$MYPATH|g" run_sft_geo.sh
sed -i "s|/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior|$MYPATH|g" merge_sft_geo.sh
sed -i "s|/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior|$MYPATH|g" run_grpo_geo.sh
sed -i "s|/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior|$MYPATH|g" LLaMA-Factory/examples/train_lora/sft_geo.yaml
sed -i "s|/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior|$MYPATH|g" LLaMA-Factory/examples/merge_lora/sft_geo_merge.yaml
sed -i "s|/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior|$MYPATH|g" open-r1-multimodal/src/open_r1/grpo_jsonl_citynav_geo.py
```

If the SFT JSON still points to old image paths, rewrite those paths too.

## Training Workflow

All commands below assume you are inside `FlightGPT_GeoPrior/` with the correct Conda environment active.

### 1. SFT

```bash
bash run_sft_geo.sh
```

Expected output:

- `saves/sft_geo/`

### 2. Merge LoRA Adapter

```bash
bash merge_sft_geo.sh
```

Expected output:

- `saves/sft_geo_merged/`

### 3. GRPO

Free the GPU first if needed, then run:

```bash
bash run_grpo_geo.sh
```

Expected output:

- `saves/grpo_geo/`

## Serving the Trained Model

Evaluation scripts call a vLLM-compatible OpenAI endpoint. Start a server after GRPO training:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ./saves/grpo_geo \
  --dtype auto \
  --trust-remote-code \
  --served-model-name qwen_2_5_vl_7b \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --port 8000 \
  --max-model-len 32000 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

Quick check:

```bash
curl http://127.0.0.1:8000/v1/models
```

## Evaluation and Demo

Full evaluation:

```bash
python eval_geo_trained.py
```

Live demo on random episodes:

```bash
python live_demo.py
```

Notes:

- `eval_geo_trained.py` and `live_demo.py` expect the vLLM server on port `8000`
- both scripts currently use `http://0.0.0.0:8000/v1` internally
- evaluation is slow; the original project notes estimate about 24 hours for the full run

## Practical Caveats

- VRAM requirement is high. The repo notes assume an RTX 4090 24 GB or similar.
- The scripts are tuned for a single-GPU workflow.
- The repository mixes project code with large-data expectations, so cloning alone is not enough to reproduce training.
- Some commands in the original workflow use `screen` for long-running jobs. That is operationally reasonable but not required.

## Recommended Next Improvements

If you want this repo to read like a strong public release, the next highest-value changes are:

1. Remove hardcoded absolute paths from scripts and YAML files.
2. Add a small bootstrap script that validates required assets before training.
3. Document where to obtain each missing artifact and expected directory names.
4. Add a minimal smoke test for prompt construction and landmark-prior lookup.
5. Move server-specific instructions out of the main README into a separate operations note.

## Citation

If you use the underlying FlightGPT work, cite the original paper:

```bibtex
@article{cai2025flightgpt,
  title={FlightGPT: Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with Vision-Language Models},
  author={Cai, Hengxing and Dong, Jinhan and Tan, Jingjun and Deng, Jingcheng and Li, Sihang and Gao, Zhifeng and Wang, Haidong and Su, Zicheng and Sumalee, Agachai and Zhong, Renxin},
  journal={arXiv preprint arXiv:2505.12835},
  year={2025}
}
```
