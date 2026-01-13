#!/bin/bash

# Script per eseguire prova.py con MMAction2
# Attiva l'ambiente mmaction, entra nella cartella del progetto e avvia lo script

echo ">>> Attivazione ambiente mmaction..."
eval "$(conda shell.bash hook)"
conda activate mmaction

echo ">>> Entrando nella cartella del progetto..."
cd "/Users/irene/Library/CloudStorage/OneDrive-UniversitàdegliStudidiUdine/Tesi Latina/project_action_recognition" || {
    echo "Errore: cartella project_action_recognition non trovata!"
    exit 1
}

echo ">>> Avvio prova.py..."
python prova.py

echo ">>> Fatto!"
