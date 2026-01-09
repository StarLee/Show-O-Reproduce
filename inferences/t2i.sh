#!/bin/bash
#cd /home/ubuntu/codespace/llm/Show-o ; 
cpath=$(cd $(dirname "$0") && pwd)
cd $cpath/.. ;
source .venv/bin/activate
python3 inference_t2i.py config=configs/showo_demo_512x512.yaml \
batch_size=1 validation_prompts_file=validation_prompts/showoprompts.txt \
guidance_scale=5 generation_timesteps=50 \
mode='t2i'