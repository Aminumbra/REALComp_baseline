import gym
import numpy as np
import realcomp
import tqdm

from PPOAgent import PPOAgent, CNN
from MultiprocessEnv import VecNormalize
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from PIL import Image

def train_cnn(env,
              model,
              model_optimizer,
              updates=100,
              batch_size=8,
              epochs=1,
              shape_pic=(240, 320, 3),
              crop=False,
              target="orange"):

    training_pics = []
    labels = []
    losses = []

    distance = torch.nn.PairwiseDistance()

    #env.render("human")
    obs = env.reset()
    
    for i in tqdm.tqdm(range(updates * batch_size * 4)):

        if i % 20 == 0:
            action = (np.random.random(9) - 0.5) * np.pi
                                
        obs, _, _, _ = env.step(action)

        if np.random.rand() < 0.06:
            rand_x = np.random.uniform(low=-0.15, high=0.05)
            rand_y = np.random.uniform(low=-0.50, high=0.50)
            env.robot.object_poses[target] = [rand_x, rand_y, 0.55, 0.00, 0.00, 0.00]
            obs = env.reset()

            
        if i % 4 == 0:
            image = obs["retina"]

            if crop:
                image = image[60:185, 30:285, :]
                image = Image.fromarray(image)
                image = image.resize((shape_pic[1], shape_pic[0]))  # Width, then height
            else:
                image = Image.fromarray(image)

            # plt.imshow(image)
            # plt.show()
                
            image = np.ravel(image) / 255.
            image = torch.FloatTensor(image)
            image = image.reshape((shape_pic[0], shape_pic[1], shape_pic[2]))
            image = image.permute(2, 0, 1)
            
            training_pics.append(image)

            labels.append(env.get_obj_pos(target)[0:2])

        if len(training_pics) == batch_size:
            # train
            for _ in range(epochs):
                training_pics_tensor = torch.FloatTensor(np.stack(training_pics))
                outputs = model(training_pics_tensor)
                labels_tensor = torch.FloatTensor(labels)

                loss = 10 * distance(outputs, labels_tensor).mean()

                model_optimizer.zero_grad()
                loss.backward()
                model_optimizer.step()
            
            training_pics = []
            labels = []
            losses.append(loss.item())

    return losses


def test_cnn(env,
         model,
         model_optimizer,
         n=10,
         shape_pic=(240, 320, 3),
         crop=False,
         target="orange"):
    
    for _ in range(n):

        rand_x = np.random.uniform(low=-0.15, high=0.05)
        rand_y = np.random.uniform(low=-0.50, high=0.50) #np.random.uniform(low=-0.50, high=0.50)
        env.robot.object_poses[target] = [rand_x, rand_y, 0.55, 0.00, 0.00, 0.00]
        obs = env.reset()

        action = (np.random.random(9) - 0.5) * np.pi
        
        for _ in range(20):
            obs, _, _, _ = env.step(action)

        image = obs["retina"]
        
        if crop:
            image = image[60:185, 30:285, :]
            image = Image.fromarray(image)
            image = image.resize((shape_pic[1], shape_pic[0]))  # Width, then height
        else:
            image = Image.fromarray(image)
            
        image = np.ravel(image) / 255.
        image = torch.FloatTensor(image)
        image = image.reshape((shape_pic[0], shape_pic[1], shape_pic[2]))
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        
        output = model(image).detach().numpy()
        label = env.get_obj_pos(target)[0:2]

        print("Output :\n%s\nLabel :\n%s\nDifference :\n%s\n\n" % (output, label, output - label))
        

if __name__=="__main__":

    crop = True
    if crop:
        shape_pic = (72, 144, 3)
    else:
        shape_pic = (240, 320, 3)
    
    model = CNN(shape_pic=shape_pic, size_output=2)
    model_optimizer = optim.Adam(model.parameters())
    env = gym.make("REALComp-v0")

    losses = train_cnn(env,
                       model,
                       model_optimizer,
                       updates=200,
                       shape_pic=shape_pic,
                       crop=crop)
   
    plt.plot(losses)
    plt.title("Losses as a function of steps")
    plt.show()

    test_cnn(env,
         model,
         model_optimizer,
         10,
         shape_pic=shape_pic,
         crop=crop)
