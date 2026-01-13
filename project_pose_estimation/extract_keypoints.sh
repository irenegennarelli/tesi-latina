#!/bin/bash

echo ">>> Attivazione ambiente openmmlab2..."
eval "$(conda shell.bash hook)"
conda activate openmmlab2

echo ">>> Entrando nella cartella del progetto..."
cd "/Users/irene/Library/CloudStorage/OneDrive-UniversitàdegliStudidiUdine/Tesi Latina/project_pose_estimation" || {
    echo "Errore: cartella project_pose_estimation non trovata!"
    exit 1
}

echo ">>> Avvio run_mmpose_inference.py..."
python extract_keypoints_from_video.py

echo ">>> Fatto!"
