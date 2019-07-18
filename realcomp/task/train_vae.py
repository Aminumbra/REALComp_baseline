import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import gym
import numpy as np
import realcomp
import tqdm

from MultiprocessEnv import VecNormalize, RobotVecEnv
import torch
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt

from VAE import VAE
import tensorboardX
import time, sys, os

from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from torchvision.transforms import transforms


# Initialize everything
experiment_name = time.strftime("%Y_%m_%d-%H_%M_%S")
if len(sys.argv) > 1:
    experiment_name = f'{sys.argv[1]}:{time.strftime("%Y_%m_%d-%H_%M_%S")}{os.getpid()}'
    
tensorboard = tensorboardX.SummaryWriter("vae/" + experiment_name, flush_secs=1)

vae = VAE(image_channels=3)
optimizer = optim.Adam(vae.parameters(), lr=1e-4)


# Define functions
def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    BCE *= 1e-5
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    KLD *= 1e4

    return BCE + KLD, BCE, KLD

# Training functions 
def collect_pictures(n=10000,
                     crop=True,
                     path="pictures/pictures/",
                     freq_images=5):

    env = gym.make('REALComp-v0')
    obs = env.reset()
    count_images = 0
    bar = tqdm.tqdm(total=n)

    i = 0

    if crop:
        shape_pic = (126, 126, 3)
    else:
        shape_pic = (240, 320, 3)

    while count_images < n:
        if i > 0 and i % 27 == 0:
            # x = np.random.uniform(-0.05, 0.05)
            # y = np.random.uniform(0.0, 0.5)
            # env.eye_pos = [x, y, 1.2]
            # env.set_eye("eye")
            obs = env.reset(mode="random")

        if i % 20 == 0:
            action = (np.random.random(9) - 0.5) * np.pi
            action[0] /= 2 # Facing the table
            action[1] = abs(action[1]) # Tends to have the arm going down
            #action = np.zeros(9)

        obs, _, _, _ = env.step(action)

        if i> 0 and i % freq_images == 0:
            image = obs["retina"]

            if crop:
                image = image[40:205, 30:285, :]
                noise = np.floor(np.random.normal(scale=0., size=image.shape))
                image = np.array(image + noise, dtype='uint8')
                image = np.clip(image, 0, 255)
                image = Image.fromarray(image)
                image = image.resize((shape_pic[1], shape_pic[0]))  # Width, then height

            else:
                image = Image.fromarray(image)

            image_name = path + "image_" + str(count_images) + ".jpeg"
            image.save(image_name)

            count_images += 1
            
            bar.update(1)

        i += 1

    bar.close()


def load_data(path="pictures/",
              batch_size=32,
              shape_pic=(126, 126, 3)):

    preprocessing = transforms.Compose([
        transforms.ToTensor()]) # Useless to use Compose for only one, but might need to add more transformations there

    train_dataset = datasets.ImageFolder(root=path, transform=preprocessing)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader
    

def train(data_loader,
          epochs = 100):

    vae.train()
    len_dataset = len(data_loader.dataset)
    
    for i in range(epochs):

        print("Epoch {}/{}".format(i, epochs))
        
        train_bce = 0
        train_kld = 0
        
        for batch_idx, (images, _) in tqdm.tqdm(enumerate(data_loader)):
            recon_images, mu, logvar = vae(images)

            loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
                
            optimizer.zero_grad()
            loss.backward()

            train_bce += bce.item()
            train_kld += kld.item()
            
            optimizer.step()
    
        tensorboard.add_scalar("VAE/BCE_Loss", train_bce / len_dataset, i)
        tensorboard.add_scalar("VAE/KLD_Loss", train_kld / len_dataset, i)

def test(crop=True,
         samples=10):

    vae.eval()
    
    if crop:
        shape_pic = (126, 126, 3)
    else:
        shape_pic = (240, 320, 3)

    env = gym.make('REALComp-v0')
    obs = env.reset(mode="random")

    for i in tqdm.tqdm(range(10 * samples)):

        if i > 0 and i % 10 == 0:
            # x = np.random.uniform(-0.05, 0.6)
            # y = np.random.uniform(-0.2, 0.2)
            # env.eye_pos = [x, y, 1.2]
            # env.set_eye("eye")
            obs = env.reset(mode="random")

        if i % 20 == 0:
            action = (np.random.random(9) - 0.5) * np.pi
            #action = np.zeros(9)
                                
        obs, _, _, _ = env.step(action)

        if i % 10 == 0:
            image = obs["retina"]

            if crop:
                image = image[40:205, 30:285, :]
                noise = np.floor(np.random.normal(scale=0., size=image.shape))
                image = np.array(image + noise, dtype='uint8')
                image = np.clip(image, 0, 255)
                image = Image.fromarray(image)
                image = image.resize((shape_pic[1], shape_pic[0]))  # Width, then height
            else:
                image = Image.fromarray(image)

            plt.imshow(image)
            plt.title("Initial image")
            plt.show()
                
            image = np.ravel(image) / 255.
            image = torch.FloatTensor(image)
            image = image.reshape((shape_pic[0], shape_pic[1], shape_pic[2]))
            image = image.permute(2, 0, 1)

            recon_image, mu, logvar = vae(image.unsqueeze(0))

            recon_image = recon_image.detach()
            recon_image = recon_image.squeeze(0)
            recon_image = recon_image.permute(1, 2, 0)
            
            plt.imshow(recon_image)
            plt.title("Reconstructed image")
            plt.show()

    ## Sampling from z :

    # dist = torch.distributions.Normal(loc=0.0, scale=1.0)

    # for i in range(5):
    #     z = torch.zeros(32)
    #     for i in range(32):
    #         z[i] = dist.sample()

    #     recon_image = vae.decode(z)

    #     recon_image = recon_image.detach()
    #     recon_image = recon_image.squeeze(0)
    #     recon_image = recon_image.permute(1, 2, 0)
        
    #     plt.imshow(recon_image)
    #     plt.title("Generated image")
    #     plt.show()

if __name__=="__main__":

    try:
        #collect_pictures(n=10000)
        train_loader = load_data()
        train(data_loader=train_loader, epochs=100)
        tensorboard.close()
        torch.save(vae.state_dict(), "vae_trained.pth")
        test()

    except KeyboardInterrupt:
        tensorboard.close()
        test()
