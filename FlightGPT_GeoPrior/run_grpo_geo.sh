#!/bin/bash
# GRPO Training with Geographic Prior
# IMPORTANT: Run run_sft_geo.sh and merge_sft_geo.sh FIRST
# Then run this script

cd /home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior/open-r1-multimodal

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true
export DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MODE="true"

RUN_NAME="FlightGPT-GeoPrior-GRPO"
export LOG_PATH="/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior/logs/debug_${RUN_NAME}.txt"

mkdir -p /home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior/logs
mkdir -p /home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior/saves/grpo_geo

torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12347" \
    src/open_r1/grpo_jsonl_citynav_geo.py \
    --output_dir /home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior/saves/grpo_geo \
    --model_name_or_path /home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior/saves/sft_geo_merged \
    --dataset_name /home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior/data/training_data/citynav_rl_data_geo.json \
    --image_folders /home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior/R1PhotoData \
    --reward_funcs accuracy format \
    --max_prompt_length 4096 \
    --max_pixels 1003520 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing false \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --report_to none
