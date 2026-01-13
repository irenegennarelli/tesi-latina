Passaggi da seguire per estrarre skeleton: 
(fuori dalla cartella project_pose_estimation)

# Installazione mmpose
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose

# Creazione conda env
./setup_openmmlab2.sh

# Estrazione skeletons per visualizzazione
./run_mmpose.sh

# Oppure estrazione skeletons per salvataggio keypoints in .json
./extract_keypoints.sh



