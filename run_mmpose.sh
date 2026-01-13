#!/bin/bash

# Script per eseguire test_script.py con MMPose
# Attiva l'ambiente openmmlab2, entra nella cartella mmpose locale e avvia lo script

echo ">>> Attivazione ambiente openmmlab2..."
eval "$(conda shell.bash hook)"
conda activate openmmlab2

echo ">>> Entrando nella cartella mmpose..."
cd "/Users/irene/Library/CloudStorage/OneDrive-UniversitàdegliStudidiUdine/Tesi Latina/mmpose" || {
    echo "Errore: cartella mmpose non trovata!"
    exit 1
}

echo ">>> Avvio test_script.py..."
python test_script.py

echo ">>> Fatto!"
