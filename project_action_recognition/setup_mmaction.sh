#!/bin/bash

# Create conda env
conda create -y -n mmaction python=3.10
conda activate mmaction

# Core deps
pip install numpy==1.23.5
pip install opencv-python==4.5.5.64 opencv-contrib-python==4.5.5.64
pip install pillow matplotlib scipy tqdm einops

# Compatible MMEngine + MMCV
pip install mmengine==0.1.0
pip install mmcv==1.5.0

# Install mmaction2 0.23.0 (compatible with mmcv 1.x)
pip install mmaction2==0.23.0

echo "Installation complete."

