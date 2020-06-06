This is a repository for Psych 229 reinforcement learning class. 

### Requirement
Python 3.7
Pytorch 
Gym (https://github.com/KarlXing/gym)
Ray (https://github.com/ray-project/ray)
Opencv-python
Numpy
Argparse


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


### Training
1. Train VAE/Class-Distentangled-VAE
```
python train_vae.py --base-dir data/  --num-splitted 1
python train_vae_disentangled.py  --base-dir data/  --num-splitted 1
```

2. Train RL
```
python main_rllib.py --encoder-path checkpoints/vae_encoder.pt
python main_rllib.py --encoder-path checkpoints/disvae_encoder.pt
```