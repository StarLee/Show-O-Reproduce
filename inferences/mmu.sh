#!/bin/bash
#cd /home/ubuntu/codespace/llm/Show-o ; 
cpath=$(cd $(dirname "$0") && pwd)
cd $cpath/.. ;
source .venv/bin/activate
python3 inference_mmu.py config=configs/showo_demo_512x512.yaml \
max_new_tokens=100 \
mmu_image_root=./mmu_validation question='Please describe this image in detail. *** Do you think the image is unusual or not?'
