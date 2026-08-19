
# Geographic Prior Injection for UAV Vision-Language Navigation
## SFT + GRPO Training with CityRefer Spatial Priors on CityNav

**Course:** CS776 — Deep Learning for Computer Vision | IIT Kanpur
**Group 1:** Kunal Jolly Saxena, Shrikant Sharma, Seetaramayya Lavu, Ankit Kumar, Harsh Sanjay Pandey, Anuj Singh, Shikha Yadav

---

## What This Project Does

We extend FlightGPT (EMNLP 2025) by injecting CityRefer landmark coordinates into
training prompts. The model is trained (SFT + GRPO) to use landmark pixel coordinates
as a search reference rather than a final answer, improving navigation precision.

**Results (5311 test episodes):**

| Split   | SR     | OSR    | NE      | SPL    |
|---------|--------|--------|---------|--------|
| Easy    | 22.99% | 42.41% | 55.20m  | 20.28% |
| Medium  | 19.93% | 35.47% | 69.25m  | 18.39% |
| Hard    | 22.56% | 34.27% | 77.68m  | 21.30% |
| Overall | 21.90% | 37.86% | 66.14m  | 19.97% |

Baseline (FlightGPT GRPO, no prior): SR=21.20%, NE=76.20m

---

## How Geographic Prior Injection Works

For every training and evaluation episode:

1. Extract landmark name: ep.description_landmarks -> ['Leslie Road']
2. Look up in CityRefer database: data/cityrefer/objects.json
3. Convert GPS to pixel: px = (world - top_left) / px_size
4. Inject into prompt before the image tag:
   [Geographic Prior] 'Leslie Road' is at map pixel [1054, 228].
   The target building is located NEAR this landmark -
   search within ~400 pixels of this location.
   <image>
5. GRPO reward: predicting [1054, 228] gives Rgoal~0.
   Predicting true target gives Rgoal=1.
   Model learns: prior = search region, not the answer.

Coverage: 90% of SFT samples, 94% of GRPO samples have a prior.

---

## Project File Structure

```
FlightGPT_GeoPrior/
├── run_sft_geo.sh                   <- Step 2: Launch SFT training
├── merge_sft_geo.sh                 <- Step 3: Merge LoRA into full model
├── run_grpo_geo.sh                  <- Step 4: Launch GRPO training
├── eval_geo_trained.py              <- Step 6: Evaluate trained model
├── eval.py                          <- Same eval with geo prior (baseline use)
├── live_demo.py                     <- Live inference demo (6 random episodes)
├── navgym/agents/
│   ├── CityNavAgent.py              <- MODIFIED: geographic_prior parameter added
│   ├── SubgoalTracker.py            <- NEW: passive subgoal tracking
│   └── VSV.py                       <- NEW: CLIP visual verification (passive)
├── data/
│   ├── cityrefer/objects.json       <- Landmark GPS database (22MB, 34 blocks)
│   ├── citynav/                     <- All CityNav JSON splits
│   └── training_data/
│       ├── citynav_rl_data_geo.json <- GRPO training data (4758 samples)
│       └── map_params.json          <- Coordinate params for all 29 blocks
├── LLaMA-Factory/
│   ├── data/
│   │   ├── vghpm_sft_v4_geo.json   <- SFT training data (4757 samples)
│   │   └── dataset_info.json       <- MODIFIED: geo dataset registered
│   └── examples/
│       ├── train_lora/sft_geo.yaml
│       └── merge_lora/sft_geo_merge.yaml
├── open-r1-multimodal/src/open_r1/
│   ├── grpo_jsonl_citynav_geo.py   <- MODIFIED: geo prior in GRPO loop
│   └── utils/                      <- Required utilities
├── model_weight/                   <- FlightGPT GRPO base model (16GB)
├── R1PhotoData/                    <- Satellite map images for evaluation (28GB)
└── saves/
    ├── sft_geo/                    <- Created after Step 2 (LoRA adapter)
    ├── sft_geo_merged/             <- Created after Step 3 (full model ~16GB)
    └── grpo_geo/                   <- Created after Step 4 (final trained model)
```

---

## PART 1 — Replicating on a Pre-Configured Server

All data, code, dependencies, and model weights are assumed to be already set up on the server.
You only need to run the commands below in order.

**Server Details:**
- IP: `<SERVER_IP>`
- User: `<USERNAME>`
- GPU: NVIDIA RTX 4090 24GB (or equivalent with ≥24GB VRAM)
- Conda environment: `dlcv_vghpm`
- Working directory: `<PATH_TO>/FlightGPT_GeoPrior/`

### Step 1 — Connect to Server

```bash
ssh <USERNAME>@<SERVER_IP>
conda activate dlcv_vghpm
cd <PATH_TO>/FlightGPT_GeoPrior/
```

### Step 2 — SFT Training (approximately 4 hours)

Trains a LoRA adapter (rank 64) on top of the FlightGPT GRPO base model using
4757 training samples enriched with geographic priors.

Open a screen session so it keeps running after you disconnect:

```bash
screen -S sft_geo
conda activate dlcv_vghpm
cd <PATH_TO>/FlightGPT_GeoPrior/
bash run_sft_geo.sh 2>&1 | tee logs/sft_geo.log
```

Detach from screen: press `Ctrl+A` then `D`

Monitor progress:

```bash
tail -f logs/sft_geo.log
# Training is working when you see: {'loss': X.XX, 'grad_norm': ...}
```

Expected output: `saves/sft_geo/` (LoRA adapter checkpoint)

> **IMPORTANT:** Do not start Step 3 until Step 2 finishes completely.
> Check completion: `grep "Training completed" logs/sft_geo.log`

### Step 3 — Merge LoRA into Full Model (approximately 10 minutes)

Fuses the LoRA adapter weights into the full model for GRPO training.

```bash
screen -S merge_geo
conda activate dlcv_vghpm
cd <PATH_TO>/FlightGPT_GeoPrior/
bash merge_sft_geo.sh 2>&1 | tee logs/merge_geo.log
```

Verify merge completed successfully:

```bash
ls -lh saves/sft_geo_merged/
# Should show multiple .safetensors files totalling ~16GB
```

### Step 4 — GRPO Training (approximately 25-40 hours)

> **IMPORTANT:** Only run after Step 3 completes and `saves/sft_geo_merged/` exists.
> GPU must be completely free. Kill any running processes first:

```bash
kill $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) 2>/dev/null
sleep 3
nvidia-smi  # verify no processes shown
```

Then start GRPO:

```bash
screen -S grpo_geo
conda activate dlcv_vghpm
cd <PATH_TO>/FlightGPT_GeoPrior/
bash run_grpo_geo.sh 2>&1 | tee logs/grpo_geo.log
```

Detach: `Ctrl+A` then `D`

Monitor progress:

```bash
tail -f logs/grpo_geo.log
tail -f logs/debug_FlightGPT-GeoPrior-GRPO.txt
```

Expected output: `saves/grpo_geo/` (final trained model)

### Step 5 — Start vLLM Server with the Trained Model

> **IMPORTANT:** GPU must be free before starting. Kill any training processes first:

```bash
kill $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) 2>/dev/null
sleep 3
nvidia-smi  # verify GPU is free
```

Start the server:

```bash
screen -S vllm_geo
conda activate dlcv_vghpm

CUDA_VISIBLE_DEVICES=0 vllm serve \
    <PATH_TO>/FlightGPT_GeoPrior/saves/grpo_geo \
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

Wait until you see: `INFO: Application startup complete.`
Detach: `Ctrl+A` then `D`

Verify server is running:

```bash
curl http://0.0.0.0:8000/v1/models
# Expected: {"data":[{"id":"qwen_2_5_vl_7b",...}]}
```

### Step 6 — Run Evaluation (approximately 24 hours)

Open a new screen session:

```bash
screen -S eval_geo
conda activate dlcv_vghpm
cd <PATH_TO>/FlightGPT_GeoPrior/
python eval_geo_trained.py 2>&1 | tee logs/eval_geo.log
```

Detach: `Ctrl+A` then `D`

Monitor progress:

```bash
tail -f logs/eval_geo.log
grep "result:" logs/eval_geo.log
```

Expected results:

```
easy result:   SR=0.2299  OSR=0.4241  NE=55.20m  SPL=0.2028
medium result: SR=0.1993  OSR=0.3547  NE=69.25m  SPL=0.1839
hard result:   SR=0.2256  OSR=0.3427  NE=77.68m  SPL=0.2130
Overall:       SR=0.2190  OSR=0.3786  NE=66.14m  SPL=0.1997
```

### Useful Server Commands

```bash
screen -ls                    # list all running screen sessions
screen -r sft_geo             # reattach to SFT session
screen -r grpo_geo            # reattach to GRPO session
screen -r eval_geo            # reattach to eval session
nvidia-smi                    # check GPU memory usage

# Kill all GPU processes if something gets stuck:
kill $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) 2>/dev/null
```

---












## PART 2 — Replicating on a New Machine

### Hardware Requirements

- GPU: NVIDIA RTX 4090 24GB minimum (1M pixel resolution needs ~20GB VRAM)
- RAM: 32GB or more
- Storage: 200GB free (model 16GB + training images 28GB + eval images 28GB + saves ~50GB)
- OS: Ubuntu 22.04 or Ubuntu 24.04

### Step 1 — Get the Code

``` bash
git clone https://github.com/Pendulumclock/FlightGPT
cd FlightGPT
unzip FlightGPT_GeoPrior_complete.zip -d ./
cd FlightGPT_GeoPrior
```

### Step 2 — Set Up Conda Environment

``` bash
conda create -n dlcv_vghpm python=3.11
conda activate dlcv_vghpm

pip install -r requirements.txt
pip install vllm==0.4.0
pip install flash-attn --no-build-isolation
pip install openai pillow tqdm
```

# CRITICAL: do NOT install autoawq - it is incompatible with transformers 4.57.6


# If it got installed accidentally: pip uninstall autoawq -y

# Install LLaMA-Factory

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..
```

# Install open-r1-multimodal

```bash
cd open-r1-multimodal
pip install -e .
cd ..
```

### Step 3 — Download FlightGPT Model Weights (approximately 16GB)

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Pendulumclock/FlightGPT',
    local_dir='./model_weight',
    ignore_patterns=['*.md']
)
"
```

### Step 4 — Download Training Map Images (approximately 28GB)

The GRPO training script needs satellite map images in R1PhotoData/.
Follow download instructions at: https://github.com/water-cookie/citynav
Place all downloaded block folders inside R1PhotoData/

mkdir -p R1PhotoData
# e.g. R1PhotoData/birmingham_block_1_20250423.../map_0....jpg

### Step 5 — Update Hardcoded Paths

All shell scripts and configs have paths hardcoded to our server.
Update them to match your machine:

``` bash 
MYPATH=$(pwd)

sed -i "s|/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior|$MYPATH|g" run_sft_geo.sh
sed -i "s|/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior|$MYPATH|g" merge_sft_geo.sh
sed -i "s|/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior|$MYPATH|g" run_grpo_geo.sh
sed -i "s|/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior|$MYPATH|g" LLaMA-Factory/examples/train_lora/sft_geo.yaml
sed -i "s|/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior|$MYPATH|g" LLaMA-Factory/examples/merge_lora/sft_geo_merge.yaml
sed -i "s|/home/priyanka/GROUP_1_DLCV/Flight_GPT/FlightGPT_GeoPrior|$MYPATH|g" open-r1-multimodal/src/open_r1/grpo_jsonl_citynav_geo.py

echo "Paths updated to: $MYPATH"
```

### Step 6 — Update Image Paths in SFT Training Data

The SFT data has image paths pointing to our server. Update to your machine:

``` bash
python3 -c "
import json, os
data = json.load(open('LLaMA-Factory/data/vghpm_sft_v4_geo.json'))
old = '/home/priyanka/GROUP_1_DLCV/Flight_GPT/VG_HPM/data/training_data/images'
new = os.path.join(os.getcwd(), 'data/training_data/images')
for s in data:
    s['images'] = [img.replace(old, new) for img in s.get('images', [])]
json.dump(data, open('LLaMA-Factory/data/vghpm_sft_v4_geo.json', 'w'), ensure_ascii=True)
print('Updated', len(data), 'samples. New path:', new)
"
```

### Step 7 — Create Required Directories

``` bash
mkdir -p logs saves
```

### Step 8 — Run Training and Evaluation

Follow Steps 2 through 6 from PART 1 exactly.
Replace the server path in the vLLM serve command with your own path:

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

---

## Troubleshooting

| Problem | Solution |
|---|---|
| OOM error during SFT or GRPO | Kill all GPU processes first: `kill $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)` |
| `autoawq` ImportError | `pip uninstall autoawq -y` |
| Cannot find valid samples | Check `dataset_info.json` has `system_tag: system` for `vghpm_sft_v4_geo` |
| GRPO fails to load model | Check `saves/sft_geo_merged/` exists: `ls -lh saves/sft_geo_merged/` |
| vLLM connection refused | vLLM not running — repeat Step 5 |
| Eval seems stuck | Normal — each episode takes 15–18s. Check: `tail -f logs/eval_geo.log` |
| `ModuleNotFoundError` | Run `conda activate dlcv_vghpm` |
| FlashAttention-2 not installed | Normal warning — falls back to torch SDPA, no impact on results |
| SFT drops all samples | Check `dataset_info.json` has `system_tag` entry for the geo dataset |
| GRPO OOM after SFT finished | vLLM or SFT process still on GPU — kill all and restart Step 4 |