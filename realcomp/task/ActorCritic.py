## ACTOR-CRITIC NETWORK

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

# A function to randomly initialize the weights

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.1)  # Centered Gaussian law, sigma = 0.1
        nn.init.constant_(m.bias, 0.1)  # b = 0.1

        
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, init=True, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            # nn.Softmax(dim=1) #Used for discrete values !
            )

        self.num_inputs       = num_inputs
        self.num_actions      = num_actions
        self.discrete_actions = discrete_actions

        self.log_std = nn.Parameter(torch.ones(1, num_actions) * std)
        
        if init:
            self.apply(init_weights)

            
    def forward(self, x):
        value  = self.critic(x)
        
        if self.discrete_actions:
            probas = self.actor(x)
            dist = Categorical(probas) #Normalize, etc : creates a categorical proba distribution

        else:
            dist = self.actor(x)
        
        return dist, value


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, discrete_actions=False, init=True, log_std=0.0):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
            )

        self.softmax = nn.Softmax(dim=1)
        self.discrete_actions = discrete_actions

        self.log_std = nn.Parameter(torch.ones(1, num_actions) * log_std)
        
        if init:
            self.apply(init_weights)


    def forward(self, x):

        output = self.actor(x)
        
        if self.discrete_actions:
            dist = self.softmax(output)
            dist = Categorical(dist) #Normalize, etc : creates a categorical proba distribution

        else:
            # Here, 'output' has to be seen as the mean
            # of a Gaussian
            
            std   = self.log_std.exp().expand_as(output)
            dist  = Normal(output, std)

        return dist
        


    
class Critic(nn.Module):
    def __init__(self, num_inputs, init=True):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            )

        if init:
            self.apply(init_weights)


    def forward(self, x):
        return self.critic(x)
