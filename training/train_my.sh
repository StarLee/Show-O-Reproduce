#!/bin/bash
cd /home/ubuntu/codespace/llm/Show-o ; 
source /home/ubuntu/codespace/llm/Show-o/.venv/bin/activate

export DS_SKIP_CUDA_CHECK=1
accelerate launch --config_file /home/ubuntu/codespace/llm/Show-o/accelerate_configs/4_gpus_deepspeed_zero2_my.yaml \
 --main_process_port=18888 \
/home/ubuntu/codespace/llm/Show-o/training/train.py \
 config=/home/ubuntu/codespace/llm/Show-o/configs/showo_pretraining_stage1_my.yaml

