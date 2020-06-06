import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Function
from torch.distributions import Beta

from utils import reparameterize


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class FixedBeta(Beta):
    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions)
            .sum(-1)
        )


class Model(nn.Module):
    def __init__(self, train_vision=False, vision_mode='vae', dist_mode='beta', latent_size=32):
        super().__init__()
        self.train_vision = train_vision  # use trained vision or train together from pixel
        self.vision_mode = vision_mode # use vae or disentangled vae as vision
        self.dist_mode = dist_mode # use beta distribution for output or categorical
        
        # vision
        if self.vision_mode == 'vae':
            self.vision = Encoder(latent_size = latent_size)
        else:
            assert(self.train_vision == False)
            self.vision = DisentangledEncoder(content_latent_size = latent_size)

        # actor
        if self.dist_mode == 'beta':
            self.actor = BetaActor()
        else:
            self.actor = CategoricalActor()
        
        # critic
        self.critic = Critic(latent_size)


    def get_feature(self, x):
        if self.train_vision:
            return self.vision.get_feature(x)
        else:
            return self.vision.get_feature(x).detach()

    # get state value
    def get_value(self, x):
        f = self.get_feature(x)
        return self.critic(f)

    # collect info for training
    def evaluate_actions(self, x, action):
        f = self.get_feature(x)
        value = self.critic(f)
        if self.dist_mode == 'beta':
            alpha, beta = self.actor(f)
            dist = FixedBeta(alpha, beta)
        else:
            dist = FixedCategorical(logits = self.actor(f))
        action_log_probs = dist.log_probs(action.squeeze(-1))
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

    # sample action for simulation
    def act(self, x):
        f = self.get_feature(x)
        value = self.critic(f)
        if self.dist_mode == 'beta':
            dist = FixedBeta(*self.actor(f))
        else:
            dist = FixedCategorical(logits = self.actor(f))
        action = dist.sample()
        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CategoricalActor(nn.Module):
    def __init__(self, input_size = 512, action_space=15):
        super().__init__()
        self.linear = nn.Linear(input_size, action_space)

    def forward(self, x):
        probs = self.linear(x)
        return probs


class BetaActor(nn.Module):
    def __init__(self, input_size = 32, action_space=3):
        super().__init__()
        self.linear_alpha = nn.Linear(input_size, action_space)
        self.linear_beta = nn.Linear(input_size, action_space)

    def forward(self, x):
        return F.softplus(self.linear_alpha(x))+1, F.softplus(self.linear_beta(x))+1
    
class Critic(nn.Module):
    def __init__(self, input_size = 512):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


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
        self.linear_logsigma = nn.Linear(2*2*256, latent_size)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)

        return mu, logsigma

    def get_feature(self, x):
        mu, logsigma = self.forward(x)
        return mu

class DisentangledEncoder(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 32, input_channel = 3):
        super(DisentangledEncoder, self).__init__()
        self.class_latent_size = class_latent_size
        self.content_latent_size = content_latent_size

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )

        self.linear_mu = nn.Linear(2*2*256, content_latent_size)
        self.linear_logsigma = nn.Linear(2*2*256, content_latent_size)
        self.linear_classcode = nn.Linear(2*2*256, class_latent_size) 

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        classcode = self.linear_classcode(x)

        return mu, logsigma, classcode

    def get_feature(self, x):
        mu, logsigma, classcode = self.forward(x)
        return mu

class Decoder(nn.Module):
    def __init__(self, latent_size = 32, output_channel = 3):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_size, 1024)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.main(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_size = 32, img_channel = 3):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_size, img_channel)
        self.decoder = Decoder(latent_size, img_channel)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        z = reparameterize(mu, logsigma)
        recon_x = self.decoder(z)

        return mu, logsigma, recon_x


class DisentangledVAE(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 32, img_channel = 3):
        super(DisentangledVAE, self).__init__()
        self.encoder = DisentangledEncoder(class_latent_size, content_latent_size, img_channel)
        self.decoder = Decoder(class_latent_size + content_latent_size, img_channel)

    def forward(self, x):
        mu, logsigma, classcode = self.encoder(x)
        contentcode = reparameterize(mu, logsigma)
        latentcode = torch.cat([contentcode, classcode], dim=1)

        recon_x = self.decoder(latentcode)

        return mu, logsigma, classcode, recon_x

