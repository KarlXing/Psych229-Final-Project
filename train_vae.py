import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import argparse
import os

from utils import VAEDataset
from model import Encoder, Decoder, VAE


parser = argparse.ArgumentParser()
parser.add_argument('--base-dir', default='./', type=str)
parser.add_argument('--batch-size', default=10, type=int, help='the actual batch size is batch_size * 5')
parser.add_argument('--num-epochs', default=2, type=int)
parser.add_argument('--num-splitted', default=10, type=int, help='number of files that the frames are stored into for each environment')
parser.add_argument('--game', default='car', type=str)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=1, type=int, help='beta in beta vae')
parser.add_argument('--save-freq', default=1000, type=int)
# parser.add_argument('--kl-tolerance', default=0.5, type=float)

args = parser.parse_args()


def updateloader(loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loader

def vae_loss(x, mu, logsigma, recon_x, beta):
    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    kl_loss = kl_loss / torch.numel(x)
    return recon_loss + kl_loss * beta, recon_loss, kl_loss

def main():
    # create direc
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    if not os.path.exists('checkimages'):
        os.makedirs("checkimages")

    # create dataset and loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = VAEDataset(args.base_dir, args.game, args.num_splitted, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # create model
    model = VAE()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # do the training
    writer = SummaryWriter()
    batch_count = 0
    for i_epoch in range(args.num_epochs):
        for i_split in range(args.num_splitted):
            for i_batch, imgs in enumerate(loader):
                batch_count += 1
                # do training for one batch
                optimizer.zero_grad()

                imgs = imgs.reshape(-1, *imgs.shape[2:]).to(device, non_blocking=True)
                mu, logsigma, recon_imgs = model(imgs)
                loss, recon_loss, kl_loss = vae_loss(imgs, mu, logsigma, recon_imgs, args.beta)
                loss.backward()
                optimizer.step()

                # write log
                writer.add_scalar('recon_loss', recon_loss.item(), batch_count)
                writer.add_scalar('kl_loss', kl_loss.item(), batch_count)

                # save image for check and model 
                if i_batch % args.save_freq == 0:
                    print("%d Epochs, %d Splitted Data, %d Batches is Done." % (i_epoch, i_split, i_batch))
                    rand_idx = torch.randperm(imgs.shape[0]) 
                    imgs = imgs[rand_idx[:9]]
                    recon_imgs = recon_imgs[rand_idx[:9]]
                    saved_imgs = torch.cat([imgs, recon_imgs], dim=0)
                    save_image(saved_imgs, "./checkimages/%1d_%1d_%05d.png" % (i_epoch, i_split,i_batch), nrow=9)

                    torch.save(model.state_dict(), "./checkpoints/model.pt")

            # load next splitted data
            updateloader(loader, dataset)
    writer.close()

if __name__ == '__main__':
    main()
