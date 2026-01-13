## Script per eseguire l'inferenza di MMPose su un video e salvare keypoints COCO17

from mmpose.apis import MMPoseInferencer
import numpy as np
import json
import os

# === CONFIG ===
#video_path = "/Users/irene/Library/CloudStorage/OneDrive-UniversitàdegliStudidiUdine/Tesi Latina/Test_neurologici/Livignani_gait3_25062025.mp4"
video_path = "/Users/irene/Library/CloudStorage/OneDrive-UniversitàdegliStudidiUdine/Tesi Latina/Test_neurologici/Lifting.mov"

output_dir = "output/COCO17"
os.makedirs(output_dir, exist_ok=True)

json_out = os.path.join(output_dir, "keypoints_coco17_lifting.json")
npy_out = os.path.join(output_dir, "keypoints_coco17_lifting.npy")

# === INFERENCER ===
inferencer = MMPoseInferencer("human", device="cpu")  # modello COCO 17 punti

results = inferencer(video_path, show=False, vis_out_dir=output_dir)

all_keypoints = []

print(">>> Running pose estimation...")

for idx, result in enumerate(results):

    preds = result["predictions"]

    if len(preds) == 0 or len(preds[0]) == 0:
        print(f"Frame {idx}: nessuna persona trovata - salvo zeri")
        kpts = np.zeros((17,2))
    else:
        person = preds[0][0]  # prima persona
        kpts = np.array(person["keypoints"])  # shape (17,2)

    all_keypoints.append(kpts.tolist())

# === SALVATAGGIO FILES ===
with open(json_out, "w") as f:
    json.dump({"keypoints": all_keypoints}, f, indent=2)

np.save(npy_out, np.array(all_keypoints))

print(">>> Salvato!")
print("JSON:", json_out)
print("NPY:", npy_out)
