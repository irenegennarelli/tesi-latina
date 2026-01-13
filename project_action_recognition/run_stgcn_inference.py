from mmaction.apis import init_recognizer, inference_recognizer
import numpy as np
import json

CONFIG = "/Users/irene/Library/CloudStorage/OneDrive-UniversitàdegliStudidiUdine/Tesi Latina/mmaction2/configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py"
CHECKPOINT = "https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint-e7bb9653.pth"
KEYPOINT_FILE = "keypoints_stgcn_lifting.npy"
CLASSES_TXT = "ntu60_classes.txt"  # <-- il file che hai appena creato

# === CARICAMENTO KEYPOINTS ===
kp = np.load(KEYPOINT_FILE)   # (1, T, 17, 3)

num_person, T, num_joints, _ = kp.shape

# === COSTRUZIONE INPUT PER MMACTION2 ===
data = dict(
    keypoint=kp.astype("float32"),
    total_frames=T,
    img_shape=(1, 224, 224),
    original_shape=(1, 224, 224),
    start_index=0,
    modality="Pose",
    # FIX fondamentale per la pipeline (anche se non usato)
    label=-1,
)

# === COSTRUISCO MODELLO ===
model = init_recognizer(CONFIG, CHECKPOINT, device="cpu")

# === INFERENZA ===
result = inference_recognizer(model, data)

print("\n===== RISULTATI ST-GCN (raw) =====")
print(result)

# === CARICO CLASSI DA FILE TXT ===
with open(CLASSES_TXT, "r") as f:
    classes = [line.strip() for line in f if line.strip()]

print(f"\nTrovate {len(classes)} classi nel file {CLASSES_TXT}")
if len(classes) != 60:
    raise ValueError(f"Mi aspetto 60 classi NTU60, ma nel file ce ne sono {len(classes)}!")

# `result` è tipicamente una lista di tuple (label_idx, score)
# oppure un array [num_class] con gli score.
# Il tuo output precedente era del tipo: [(41, 12.18), (25, 6.84), ...]
# quindi gestiamo quel caso.

# Converto in lista di (idx, score)
if isinstance(result, (list, tuple)) and len(result) > 0 and isinstance(result[0], (list, tuple)):
    pred = result
else:
    # fallback: assumiamo che sia un vettore di score e prendiamo i top-5
    scores = np.array(result)
    top5_idx = scores.argsort()[::-1][:5]
    pred = [(int(i), float(scores[i])) for i in top5_idx]

print("\n===== TOP-5 PREDIZIONI ST-GCN (con etichette NTU60) =====")
for rank, (idx, score) in enumerate(pred[:5], start=1):
    if 0 <= idx < len(classes):
        label_name = classes[idx]
    else:
        label_name = f"[indice {idx} fuori range]"
    # idx è 0-based, ma A-xxx nel paper è 1-based -> mostro anche A{idx+1}
    print(f"{rank}. A{idx+1:02d} - {label_name}  (classe_idx={idx}, score={score:.3f})")
