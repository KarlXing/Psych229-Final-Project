This is the repository for Psych 229 reinforcement learning class. 

### Requirement
Python 3.7  
Pytorch   
Gym (https://github.com/KarlXing/gym)   
Ray (https://github.com/ray-project/ray)    
Opencv-python  
Numpy  
Argparse  

### Environment Installation
```
# create virtual environments
conda env update --name psychrl python=3.7
conda activate psychrl

# install reinforcement learning training libs
conda install pytorch torchvision cudatoolkit=10.1
pip install --upgrade pip
pip install opencv-python
pip install numpy
pip install argparse
pip install pandas
pip install ray==0.8.4
pip install ray[rllib]==0.8.4

git clone https://github.com/karlxing/gym.git
cd gym
pip install -e .[box2d]
```

### Clone This Project To The directory You Want
```
git clone https://github.com/KarlXing/Psych229-Final-Project.git
cd Psych229-Final-Project
```

### Training
1. Train VAE/Class-Distentangled-VAE  
(The collected CarRacing images are uploaded to directory data. But only a small portion of images are uploaded since the whole training dataset is too big. Two models trained with full dataset are uploaded to checkpoints directory.)
```
# Train VAE
python train_vae.py --base-dir data/  --num-splitted 1

# Train Cycle-consistent VAE (Disentangled VAE)
python train_vae_disentangled.py  --base-dir data/  --num-splitted 1
```

2. Train RL
```
# Train Reinforcement Learning With Latent Representation From VAE
python main_rllib.py --encoder-path checkpoints/vae_encoder.pt

# Train Reinforcement Learning With Latent Representation From Disentangled VAE
python main_rllib.py --encoder-path checkpoints/disvae_encoder.pt
```
