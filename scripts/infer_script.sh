#!/bin/bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
cuda_device=0
echo "Start Infer on device $cuda_device"
CUDA_VISIBLE_DEVICES=$cuda_device python infer_script.py --config ../configs/infer.config