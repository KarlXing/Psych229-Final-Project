import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog, ActionDistribution
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn
from torch.distributions import Beta, Normal
from torch.distributions.kl import kl_divergence
import gym
from gym.spaces import Box
import cv2
import numpy as np
import argparse

from model import Encoder


######## Args Setting ##########
parser = argparse.ArgumentParser()
parser.add_argument('--train-epochs', default=5000, type=int)
parser.add_argument('--trainer-save-path', default='./', type=str)
parser.add_argument('--encoder-path', default=None)
parser.add_argument('--model-save-freq', default=50, type=int)
parser.add_argument('--model-save-path', default='./', type=str)
parser.add_argument('--train-encoder', default=False, type=bool)
parser.add_argument('--num-workers', default=2, type=int)
parser.add_argument('--num-envs-per-worker', default=1, type=int)
parser.add_argument('--num-gpus', default=1, type=int)
parser.add_argument('--use-gae', default=True, type=bool)
parser.add_argument('--batch-mode', default='truncate_episodes', type=str)
parser.add_argument('--vf-loss-coeff', default=1, type=int)
parser.add_argument('--vf-clip-param', default=1000, type=int)
parser.add_argument('--reward-wrapper', default=True, type=bool, help='whether using reward wrapper so that avoid -100 penalty')
parser.add_argument('--lr', default=0.00005, type=float)
parser.add_argument('--kl-coeff', default=0, type=float)
parser.add_argument('--num-sgd-iter', default=10, type=int)
parser.add_argument('--sgd-minibatch-size', default=128, type=int)
parser.add_argument('--grad-clip', default=0.5, type=float, help='other implementations may refer as max_grad_norm')
parser.add_argument('--rollout-fragment-length', default=10, type=int)
parser.add_argument('--train-batch-size', default=2000, type=int)
parser.add_argument('--clip-param', default=0.1, type=float, help='other implementations may refer as clip_ratio')
args = parser.parse_args()


######## Env Setting ###########
def process_obs(obs): # a single frame (96, 96, 3) for CarRacing
    obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
    return np.transpose(obs, (2,0,1))

# Todo: choose carracing1/2/3/4/5 based on worker_index
def choose_env(worker_index):
    return 'CarRacing-v0'

class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make(choose_env(env_config.worker_index))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation_space = Box(low=0, high=255, shape=(3,64,64), dtype=self.env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        processed_obs =  process_obs(obs)
        return processed_obs

    def step(self, action):
        action[0] = action[0]*2-1
        obs, reward, done, info = self.env.step(action)
        processed_obs = process_obs(obs)
        return processed_obs, reward, done, info

class MyEnvRewardWrapper(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make(choose_env(env_config.worker_index))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation_space = Box(low=0, high=255, shape=(3,64,64), dtype=self.env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        processed_obs =  process_obs(obs)
        return processed_obs

    def step(self, action):
        action[0] = action[0]*2-1
        obs, reward, done, info = self.env.step(action)
        processed_obs = process_obs(obs)
        # according to the game description, there should only be -0.1 time penalty. Avoid other penalties such as crash on black boundary
        if reward < 0:
            reward = -0.1
        return processed_obs, reward, done, info

if args.reward_wrapper:
    register_env("myenv", lambda config: MyEnvRewardWrapper(config))
else:
    register_env("myenv", lambda config: MyEnv(config))


######## Model Setting ##########
class Encoder(nn.Module):
    def __init__(self, latent_size = 32, input_channel = 3):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )
        self.linear_mu = nn.Linear(2*2*256, latent_size)

    def forward(self, x):
        x = self.main(x/255.0)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        return mu

class MyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        custom_config = model_config['custom_options']

        self.main = Encoder()
    
        if custom_config['encoder_path'] is not None:
            print("Load Trained Encoder")
            # saved checkpoints could contain extra weights such as linear_logsigma 
            weights = torch.load(custom_config['encoder_path'], map_location={'cuda:0':'cpu'})
            for k in list(weights.keys()):
                if k not in self.main.state_dict().keys():
                    del weights[k]
            self.main.load_state_dict(weights)
        
        self.critic = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 1))
        self.actor = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 3), nn.Sigmoid())
        self.actor_logstd = nn.Parameter(torch.zeros(3), requires_grad=True)
        self._cur_value = None
        print("Train Encoder:", custom_config['train_encoder'])
        self.train_encoder = custom_config['train_encoder']

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        features = self.main(input_dict['obs'].float())
        if not self.train_encoder:
            features = features.detach() # not train the encoder

        actor_mu = self.actor(features) # Bx3
        batch_size = actor_mu.shape[0] 
        actor_logstd = torch.stack(batch_size * [self.actor_logstd], dim=0) # Bx3
        logits = torch.cat([actor_mu, actor_logstd], dim=1)
        self._cur_value = self.critic(features).squeeze(1)

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, 'Must call forward() first'
        return self._cur_value

ModelCatalog.register_custom_model("mymodel", MyModel)

############ Distribution Setting ##############
class MyDist(ActionDistribution):
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 6

    def __init__(self, inputs, model):
        super(MyDist, self).__init__(inputs, model)
        mu = inputs[:, :3]
        logstd = inputs[:, 3:]
        self.dist = Normal(mu, logstd.exp())

    def sample(self):
        self.sampled_action = self.dist.sample()
        return self.sampled_action

    def deterministic_sample(self):
        return self.dist.loc

    def sampled_action_logp(self):
        return self.logp(self.sampled_action)

    def logp(self, actions):
        return self.dist.log_prob(actions).sum(-1)

    # refered from https://github.com/pytorch/pytorch/blob/master/torch/distributions/kl.py
    def kl(self, other):
        p, q = self.dist, other.dist
        return kl_divergence(p, q).sum(-1)

    def entropy(self):
        return self.dist.entropy().sum(-1)



ModelCatalog.register_custom_action_dist("mydist", MyDist)


########### Do Training #################
def main():
    ray.init()
    
    #  Hyperparameters of PPO are not well tuned. Most of them refer to https://github.com/xtma/pytorch_car_caring/blob/master/train.py
    trainer = PPOTrainer(env=MyEnv, config={
        "use_pytorch": True,
        "model":{"custom_model":"mymodel", 
                "custom_options":{'encoder_path':args.encoder_path, 'train_encoder':args.train_encoder},
                "custom_action_dist":"mydist",
                },
        "env_config":{'game':'CarRacing'},
        "num_workers":args.num_workers,
        "num_envs_per_worker":args.num_envs_per_worker,
        "num_gpus":args.num_gpus,
        "use_gae":args.use_gae,
        "batch_mode":args.batch_mode,
        "vf_loss_coeff":args.vf_loss_coeff,
        "vf_clip_param":args.vf_clip_param,
        "lr":args.lr,
        "kl_coeff":args.kl_coeff,
        "num_sgd_iter":args.num_sgd_iter,
        "grad_clip":args.grad_clip,
        "clip_param":args.clip_param,
        "rollout_fragment_length":args.rollout_fragment_length,
        "train_batch_size":args.train_batch_size,
        "sgd_minibatch_size":args.sgd_minibatch_size
        })


    for i in range(args.train_epochs):
        trainer.train()
        print("%d Train Done" % (i), "Save Freq: %d" % (args.model_save_freq))
        if (i+1) % args.model_save_freq == 0:
            print("%d Episodes Done" % (i))
            weights = trainer.get_policy().get_weights()
            torch.save(weights, args.model_save_path+"%d-mode.pt" % (i+1))
    trainer.save(args.trainer_save_path)
    print("Done All!")
    trainer.stop()

if __name__ == '__main__':
    main()