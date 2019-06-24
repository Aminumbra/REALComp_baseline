import numpy as np


## CONVOLUTIONAL NEURAL NETWORK

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import torch.optim as optim
import torch.optim as optim

# A function to randomly initialize the weights

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1)  # Centered Gaussian law, sigma = 0.1
        nn.init.constant_(m.bias, 0.1)  # b = 0.1

def kernelOutputSize(input_size, kernel_size, stride, padding):
    return int((input_size - kernel_size + 2*padding) / stride) + 1


class CNN(nn.Module):
    def __init__(self,
                 height_visual_inputs    = 240,  # Our images : 240*320*3
                 width_visual_inputs     = 320,
                 channels_visual_inputs  = 3,
                 size_observation_space  = 13,   # Our robot : 9 joints + 4 sensors
                 num_actions             = 9,    # Our robot : 9 angles
                 init                    = True,
                 std                     = 0.0):
        
        super(CNN, self).__init__()
        
        self.convolutions_1 = nn.Sequential(
            nn.Conv2d(channels_visual_inputs, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
            )

        self.convolutions_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.convolutions_3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.dropout = nn.Dropout()

        self.flatten = nn.Linear(16 * 15 * 20 * 2, 128) #*2 is because we have obs AND goal

        self.other_obs_layer = nn.Linear(size_observation_space, 128)

        self.fc_1    = nn.Linear(128 + 128, 128)
        self.fc_2    = nn.Linear(128, 32)
        self.fc_3    = nn.Linear(32, num_actions)


        self.optimizer = optim.Adam(self.parameters())
        self.num_actions = num_actions

        
    def forward(self, visual_obs, others_obs, goal):

        obs_and_goal = torch.cat((visual_obs, goal))
        
        x = self.convolutions_1(obs_and_goal)

        x = self.convolutions_2(x)
        x = self.convolutions_3(x)

        x = self.dropout(x)

        x = torch.reshape(x, (-1,))
        cnn_obs_pic = self.flatten(x)
        obs_others  = self.other_obs_layer(others_obs)

        full_obs    = torch.cat((cnn_obs_pic, obs_others))
        
        out = self.fc_1(full_obs)
        out = self.fc_2(out)
        out = self.fc_3(out)

        return out


    def forward_noise(self, visual_obs, others_obs, goal, std=1.0):

        actions  = self.forward(visual_obs, others_obs, goal)
        log_std = torch.ones(1, self.num_actions) * std
        noisy_action = Normal(actions, std)

        return noisy_action.sample()



def observationToCNNInput(observation, visual_width=320, visual_height=240):
    """
    Transforms the observation of the environment into
    an object with a suitable shape for the CNN
    """

    joint_pos     = torch.FloatTensor(observation["joint_positions"])
    touch_sensors = torch.FloatTensor(observation["touch_sensors"])
    retina        = torch.FloatTensor(observation["retina"])
    retina        = torch.reshape(retina, (-1, 3, visual_height, visual_width))
    goal          = torch.FloatTensor(observation["goal"])
    goal          = torch.reshape(goal, (-1, 3, visual_height, visual_width))

    return retina, torch.cat((joint_pos, touch_sensors)), goal




###########################################################################


class TestPolicy:
    
    def __init__(self, action_space):
        self.action_space = action_space
        self.action       = np.zeros(action_space.shape[0])
        self.model        = CNN()

    def step(self, observation, reward, done):
        #self.action += 0.1*np.pi*np.random.randn(self.action_space.shape[0])

        visual_obs, other_obs, goal = observationToCNNInput(observation)

        actions = self.model.forward_noise(visual_obs, other_obs, goal)

        actions = actions.detach()
        self.action = actions
        return self.action
    

    def update(self, rewards):
        #TODO

        loss = torch.FloatTensor(rewards).detach()
        loss.requires_grad = True
        loss = loss.mean()
        
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

TestController = TestPolicy
