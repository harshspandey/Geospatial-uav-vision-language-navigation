#!/bin/bash
cd /home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior/LLaMA-Factory
export CUDA_VISIBLE_DEVICES=0
export DISABLE_VERSION_CHECK=1
conda activate dlcv_vghpm
python src/train.py examples/merge_lora/sft_geo_merge.yaml
