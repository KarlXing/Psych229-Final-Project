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

from utils import VAEDataset, reparameterize
from model import Encoder, Decoder, DisentangledVAE


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
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--class-latent-size', default=8, type=int)
parser.add_argument('--content-latent-size', default=24, type=int)
# parser.add_argument('--kl-tolerance', default=0.5, type=float)

args = parser.parse_args()


def updateloader(loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loader


def vae_loss(x, mu, logsigma, recon_x, beta=1):
    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    kl_loss = kl_loss / torch.numel(x)
    return recon_loss + kl_loss * beta


def forward_loss(x, model):
    mu, logsigma, classcode = model.encoder(x)
    contentcode = reparameterize(mu, logsigma)
    shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]

    latentcode1 = torch.cat([contentcode, shuffled_classcode], dim=1)
    latentcode2 = torch.cat([contentcode, classcode], dim=1)

    recon_x1 = model.decoder(latentcode1)
    recon_x2 = model.decoder(latentcode2)

    return vae_loss(x, mu, logsigma, recon_x1) + vae_loss(x, mu, logsigma, recon_x2)


def backward_loss(x, model, device):
    mu, logsigma, classcode = model.encoder(x)
    shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]
    randcontent = torch.randn_like(mu).to(device)

    latentcode1 = torch.cat([randcontent, classcode], dim=1)
    latentcode2 = torch.cat([randcontent, shuffled_classcode], dim=1)

    recon_imgs1 = model.decoder(latentcode1).detach()
    recon_imgs2 = model.decoder(latentcode2).detach()

    cycle_mu1, cycle_logsigma1, cycle_classcode1 = model.encoder(recon_imgs1)
    cycle_mu2, cycle_logsigma2, cycle_classcode2 = model.encoder(recon_imgs2)

    cycle_contentcode1 = reparameterize(cycle_mu1, cycle_logsigma1)
    cycle_contentcode2 = reparameterize(cycle_mu2, cycle_logsigma2)

    bloss = F.l1_loss(cycle_contentcode1, cycle_contentcode2)
    return bloss


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
    model = DisentangledVAE(class_latent_size = args.class_latent_size, content_latent_size = args.content_latent_size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # forward_optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # backward_optimizer = optim.Adam(model.encoder.parameters(), lr=args.learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # do the training
    writer = SummaryWriter()
    batch_count = 0
    for i_epoch in range(args.num_epochs):
        for i_split in range(args.num_splitted):
            for i_batch, imgs in enumerate(loader):
                batch_count += 1
                # forward circle
                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True)
                # forward_optimizer.zero_grad()
                optimizer.zero_grad()

                floss = 0
                for i_class in range(imgs.shape[0]):
                    image = imgs[i_class]
                    floss += forward_loss(image, model)
                floss = floss / imgs.shape[0]

                # floss.backward()
                # forward_optimizer.step()

                # backward circle
                imgs = imgs.reshape(-1, *imgs.shape[2:])
                # backward_optimizer.zero_grad()

                bloss = backward_loss(imgs, model, device)

                # bloss.backward()
                # backward_optimizer.step()
                (floss + bloss * args.bloss_coef).backward()
                optimizer.step()

                # write log
                writer.add_scalar('floss', floss.item(), batch_count)
                writer.add_scalar('bloss', bloss.item(), batch_count)

                # save image for check and model 
                if i_batch % args.save_freq == 0:
                    print("%d Epochs, %d Splitted Data, %d Batches is Done." % (i_epoch, i_split, i_batch))
                    rand_idx = torch.randperm(imgs.shape[0])
                    imgs1 = imgs[rand_idx[:9]]
                    imgs2 = imgs[rand_idx[-9:]]
                    with torch.no_grad():
                        mu, _, classcode1 = model.encoder(imgs1)
                        _, _, classcode2 = model.encoder(imgs2)
                        recon_imgs1 = model.decoder(torch.cat([mu, classcode1], dim=1))
                        recon_combined = model.decoder(torch.cat([mu, classcode2], dim=1))

                    saved_imgs = torch.cat([imgs1, imgs2, recon_imgs1, recon_combined], dim=0)
                    save_image(saved_imgs, "./checkimages/%1d_%1d_%05d.png" % (i_epoch, i_split,i_batch), nrow=9)

                    torch.save(model.state_dict(), "./checkpoints/model.pt")

            # load next splitted data
            updateloader(loader, dataset)
    writer.close()

if __name__ == '__main__':
    main()
