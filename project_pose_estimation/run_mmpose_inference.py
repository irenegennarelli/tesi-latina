## Script per testare l'inferenza di MMPose (wholebody) su un video e salvare i keypoint in un file JSON

from mmpose.apis import MMPoseInferencer
import os
#import json

video_path = '/Users/irene/Library/CloudStorage/OneDrive-UniversitàdegliStudidiUdine/Tesi Latina/Test_neurologici/Testa_multitask_031225.mp4'
os.makedirs('outputs', exist_ok=True)
output_path = 'output/Wholebody'
#keypoint_json = os.path.join(output_path, "keypoints.json")

assert os.path.exists(video_path), f"Video non trovato: {video_path}"

inferencer = MMPoseInferencer('wholebody', device='cpu')

# Lista dove salveremo i risultati
#all_frames = []

result_generator = inferencer(
    video_path,
    show=False,
    radius=4, thickness=2,
    #vis_out_dir='outputs',   
    vis_out_dir=output_path,
    #device='cpu'
)

for _ in result_generator:
    pass
print("✅ Inference terminata")

# print(">>> Inizio estrazione keypoint...")

# for frame_id, result in enumerate(result_generator):

#     preds = result['predictions'][0]  # primo (e unico) video

#     persons = []
#     if 'keypoints' in preds:
#         for i in range(len(preds['keypoints'])):
#             persons.append({
#                 "keypoints": preds['keypoints'][i].tolist(),
#                 "scores": preds['keypoint_scores'][i].tolist()
#             })

#     all_frames.append({
#         "frame_id": frame_id,
#         "persons": persons
#     })

# # Salva il JSON
# with open(keypoint_json, "w") as f:
#     json.dump(all_frames, f, indent=4)

# print(f"✅ Inference terminata. Keypoint salvati in:\n{keypoint_json}")
