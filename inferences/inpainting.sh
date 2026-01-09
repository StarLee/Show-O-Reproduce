#!/bin/bash
#cd /home/ubuntu/codespace/llm/Show-o ; 
cpath=$(cd $(dirname "$0") && pwd)
cd $cpath/.. ;
source .venv/bin/activate
python3 inference_t2i.py config=configs/showo_demo.yaml \
batch_size=1 \
guidance_scale=1.75 generation_timesteps=16 \
mode='inpainting' prompt='A blue sports car with sleek curves and tinted windows, parked on a bustling city street.' \
image_path=./inpainting_validation/bus.jpg inpainting_mask_path=./inpainting_validation/bus_mask.webp