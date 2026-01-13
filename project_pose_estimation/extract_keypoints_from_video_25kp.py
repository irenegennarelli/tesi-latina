from mmpose.apis import MMPoseInferencer
import numpy as np
import json
import os

# === CONFIG ===
#video_path = "/Users/irene/Library/CloudStorage/OneDrive-UniversitàdegliStudidiUdine/Tesi Latina/Test_neurologici/Livignani_gait3_25062025.mp4"
video_path = "/Users/irene/Library/CloudStorage/OneDrive-UniversitàdegliStudidiUdine/Tesi Latina/Test_neurologici/Lifting.mov"

output_dir = "output/COCO25"
os.makedirs(output_dir, exist_ok=True)

json_out = os.path.join(output_dir, "keypoints_lifting.json")
npy_out = os.path.join(output_dir, "keypoints_lifting.npy")

# === MODELLO POSE: COCO BODY25 ===
inferencer = MMPoseInferencer("rtmpose-m-body25", device="cpu")

results = inferencer(video_path, show=False, vis_out_dir=output_dir)

all_kp = []

print(">>> Estrazione keypoints body25...")

for idx, result in enumerate(results):
    pred = result["predictions"][0]

    if len(pred["keypoints"]) == 0:
        key = np.zeros((25, 2))
    else:
        key = np.array(pred["keypoints"][0])[:, :2]  # (25,2)

    all_kp.append(key.tolist())

# === SALVATAGGIO ===
with open(json_out, "w") as f:
    json.dump({"keypoints": all_kp}, f)

np.save(npy_out, np.array(all_kp))

print(">>> Salvati keypoints in:")
print("-", json_out)
print("-", npy_out)
