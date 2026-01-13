# Script per testare un video con modello SlowFast preaddestrato su Kinetics400
# nell'environment mmaction nella cartella mmaction2
# Modello pesante, ma accurato. Usa CPU.
# Inference su video di test (gait): Cecconi_gait_18.07.25.mp4
# Risultati:
#tap dancing (349): 0.110
#playing trumpet (248): 0.050
#pushing car (261): 0.048
#pull ups (255): 0.046
#playing trombone (247): 0.037
# RGB-based non va bene per i video di cammino neurologico (troppo brevi)
# Usare skeleton-based (mmpose + action recognition su sequenza di skeleton)

from mmaction.apis import init_recognizer, inference_recognizer

# CONFIG + CHECKPOINT
#config = 'configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py'
config = '/Users/irene/Library/CloudStorage/OneDrive-UniversitàdegliStudidiUdine/Tesi Latina/mmaction2/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py'
checkpoint = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth' 

# QUI metti il tuo video:
video = '/Users/irene/Library/CloudStorage/OneDrive-UniversitàdegliStudidiUdine/Tesi Latina/Test_neurologici/Cecconi_gait_18.07.25.mp4'

# build model
model = init_recognizer(config, checkpoint, device='cpu')

# run inference
result = inference_recognizer(model, video)

# load label map
label_map = 'tools/data/kinetics/label_map_k400.txt'
labels = [x.strip() for x in open(label_map)]

# print results
print("\nRISULTATI INFERENZA:")
for idx, score in result:
    print(f"{labels[idx]} ({idx}): {score:.3f}")
