import json
import numpy as np
import os

INPUT_JSON = "../project_pose_estimation/output/COCO17/keypoints_coco17_lifting.json"
OUTPUT_NPY = "keypoints_stgcn_lifting.npy"

NUM_KEYPOINTS = 17

# === CARICAMENTO ===
with open(INPUT_JSON, "r") as f:
    data = json.load(f)["keypoints"]   # (frames, 17, 2)

data = np.array(data)  # (T, 17, 2)

num_frames = data.shape[0]

# Creiamo array ST-GCN: (T, 1, 17, 3)
skeleton = np.zeros((num_frames, 1, NUM_KEYPOINTS, 3), dtype=np.float32)

# Inseriamo le coordinate
skeleton[:, 0, :, :2] = data     # x,y
skeleton[:, 0, :, 2] = 1.0       # confidence fittizia

# === RIORDINAMENTO PER MMAction2 ===
# Da (T, 1, 17, 3) → (1, T, 17, 3)
skeleton = skeleton.transpose(1, 0, 2, 3)

# === SALVATAGGIO ===
np.save(OUTPUT_NPY, skeleton)

print(f"Creato file ST-GCN: {OUTPUT_NPY}")
print(f"Shape finale (1, T, 17, 3): {skeleton.shape}")
