#!/bin/bash
cd /home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior/LLaMA-Factory
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true
export DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda activate dlcv_vghpm
python src/train.py examples/train_lora/sft_geo.yaml
