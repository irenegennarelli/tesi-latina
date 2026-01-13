#!/bin/bash
set -e

echo ">>> CREAZIONE AMBIENTE openmmlab2..."
conda create --name openmmlab2 python=3.8 -y

echo ">>> Attivo ambiente..."
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate openmmlab2

echo ">>> Installo PyTorch CPU (compatibile MAC ARM)..."
conda install pytorch torchvision cpuonly -c pytorch -y

echo ">>> Installo openmim..."
pip install -U openmim

echo ">>> Installo mmengine..."
mim install mmengine

echo ">>> Installo mmcv>=2.0.1..."
mim install "mmcv>=2.0.1"

echo ">>> Correggo versioni incompatibili... (mmcv e mmdet)"
pip uninstall -y mmdet mmcv

echo ">>> Installo mmcv 2.1.0..."
mim install "mmcv==2.1.0"

echo ">>> Installo mmdet 3.2.0..."
mim install "mmdet==3.2.0"

echo ">>> Installo librerie generiche..."
pip install numpy opencv-python pillow matplotlib scipy torchvision

echo ">>> Creo fake xtcocotools per compatibilità..."
XT_PATH="/opt/anaconda3/envs/openmmlab2/lib/python3.8/site-packages/xtcocotools"
mkdir -p "$XT_PATH"
echo "from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask" > "$XT_PATH/__init__.py"

echo ">>> Installo MMPose 1.3.2 dalla cartella locale..."
cd "/Users/irene/Library/CloudStorage/OneDrive-UniversitàdegliStudidiUdine/Tesi Latina/mmpose"
pip install -r requirements.txt
pip install -e .

echo ">>> FATTO! Ambiente openmmlab2 configurato."
