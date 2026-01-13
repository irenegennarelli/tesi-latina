conda create -n skelmamba python=3.10 -y
conda activate skelmamba

pip install wandb weave
wandb login

git clone https://github.com/iN1k1/gait-uniroma1          

cd gait-uniroma1

pip install -r requirements.txt

pip install -e .

import os
os.environ['WANDB_API_KEY'] = '2e38c245dea31fa8971c02fdd0fff5d176612465'