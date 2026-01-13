#!/bin/bash

echo ">>> Attivazione ambiente mmaction..."
eval "$(conda shell.bash hook)"
conda activate mmaction

echo ">>> Entrando in project_action_recognition..."
cd "/Users/irene/Library/CloudStorage/OneDrive-UniversitàdegliStudidiUdine/Tesi Latina/project_action_recognition" || {
    echo "Errore: cartella del progetto non trovata!"
    exit 1
}

echo ">>> Conversione keypoints -> ST-GCN..."
python convert_keypoints_to_stgcn.py

echo ">>> Inferenza ST-GCN..."
python run_stgcn_inference.py

echo ">>> Finito!"
