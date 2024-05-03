# ConstructionStageClassificationPytorchLightning
ConstructionStageClassification using PytorchLightning
## ToDo
1. Building models using Pytorch lighting (done)
2. Reading dataset (done), evaluation metrics using scikit learn (done)
3. git branch, gitignore, git merge (done)
4. using logging (done)
5. using wandb (done)
6. load checkpoints and test
## Setup
conda create --name pltenv python=3.12,1
conda activate pltenv
pip install requirements.txt
## Training
python main_pl.py
## Testing
python test_pl.py --idx 0