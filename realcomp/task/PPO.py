import gym
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from Utils import plot
from matplotlib import pyplot as plt

from MultiprocessEnv import SubprocVecEnv

from ActorCritic import ActorCritic, Actor, Critic
from CNN import CNN

# Paper : https://arxiv.org/pdf/1707.06347.pdf

## CUDA

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

## ENV

def make_env(env_id):
        def _thunk():
            return gym.make(env_id)

        return _thunk
    
def create_envs(env_id="Pendulum-v0", num_envs=16):
    ##currently working environment : Pendulum-v0, BipedalWalker-v2
    #Continuous action space, continous observation space
    
    env = gym.make(env_id) # current, testing env

    discrete_action_space = isinstance(env.action_space, gym.spaces.discrete.Discrete)

    envs = [make_env(env_id) for e in range(num_envs)]
    envs = SubprocVecEnv(envs) # Wrapper simulating a threading situation, 1 env/Thread

    size_obs = envs.observation_space.shape[0]

    if discrete_action_space:
        num_actions = envs.action_space.n
    else:
        num_actions = envs.action_space.shape[0]

    print("Env name : ", env_id)
    print("Different elements per observation : ", size_obs)
    print("Number of actions : ", num_actions)
    print("Simulated environments : ", num_envs)

    return env, envs, size_obs, num_actions, discrete_action_space

# Run the function now, as several things are global variables
# used later in the code

env, envs, size_obs, num_actions, discrete_action_space = create_envs(env_id="BipedalWalker-v2", num_envs=16)

########################################################

## MODEL

# Model hyperparameters

lr = 3e-4

#model = ActorCritic(size_obs, num_actions).to(cpu) # Single network
model_actor      = Actor(size_obs, num_actions, discrete_actions=discrete_action_space).to(device)
model_critic     = Critic(size_obs).to(device)
optimizer_actor  = optim.Adam(model_actor.parameters(),  lr=lr)
optimizer_critic = optim.Adam(model_critic.parameters(), lr=lr)
    
def test_env(render=False):
    state = env.reset()

    if render:
        env.render()

    done = False
    tot_reward = 0

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        #dist, _ = model(state)
        dist  = model_actor(state)
        
        next_state, reward, done, info = env.step(dist.sample().cpu().numpy()[0])
        
        state = next_state

        tot_reward += reward

        if render:
            env.render()

    return tot_reward


# Article, page 5
# The article's 'lambda' is here a 'tau'
# Based on GAE
# We do not return the advantages, but Values + Advantages, i.e. the estimated return

# not_terminal_states is used to place 0-reward for terminal states
def compute_returns(next_value, rewards, not_terminal_states, values, gamma=0.99, tau=0.95):
    
    values = values + [next_value] #Can't simply append, as it would modify external values

    advantage = 0
    returns   = []

    for step in reversed(range(len(rewards))):

        delta     = rewards[step] + gamma * values[step + 1] * not_terminal_states[step] - values[step]
        advantage = delta + gamma * tau * not_terminal_states[step] * advantage
        
        returns.insert(0, advantage + values[step])

    return returns


def compute_goal_reward(observation):
        ###TODO

        ## Should give a reward measuring how far we are from the fixed goal

        ## Ideas :
        ## Sparse : 1 if matching, 0 otherwise
        ## Handcrafted : MSE(obs, goal)
        ## Other things are possible ...

        return 0


def ppo_iterator(mini_batch_size, states, actions, log_probas, returns, advantages):

    n_states = states.size(0)

    for _ in range(n_states // mini_batch_size):
        # generates mini_batch_size indices
        indices = np.random.randint(0, n_states, mini_batch_size)
        
        yield states[indices, :], \
            actions[indices,:],   \
            log_probas[indices,:],\
            returns[indices,:],   \
            advantages[indices,:]


def ppo_full_step(ppo_epochs,
                  mini_batch_size,
                  states,
                  actions,
                  log_probas,
                  returns,
                  advantages,
                  clip=0.2,
                  entropy_coeff=0.005):
        
    for _ in range(ppo_epochs):

        for state, action, old_log_probas, return_, advantage in ppo_iterator(mini_batch_size, states, actions, log_probas, returns, advantages):

            #dist, value = model(state)
            dist  = model_actor(state)
            
            value = model_critic(state)

            entropy = dist.entropy().mean()
            
            new_log_probas = dist.log_prob(action)

            ratio = (new_log_probas - old_log_probas).exp() #Not simply new/old, as we have LOG of them

            estimate_1 = ratio * advantage
            estimate_2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage

            #L_CLIP in the paper

            ## TODO : should take into account the "Goal Reward"
            actor_loss  = - torch.min(estimate_1, estimate_2).mean() # We consider the opposite, as we perform gradient ASCENT

            if discrete_action_space:
                    actor_loss -= entropy_coeff * entropy
                    
            #L_VF in the paper
            critic_loss = ((return_ - value) ** 2).mean()

            # Loss determined by equation (9) from the paper
            # 0.5 and 0.001 parameters arbitrary
            #loss = -(actor_loss - 0.5*critic_loss + 0.001*entropy)

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()


## TRAINING

num_frames = 50000
test_rewards = []
epochs = 4
mini_batch_size = 5
num_steps = 20

def training(plot_final=True, num_frames=num_frames):

    state = envs.reset() # np.stack of all the environments "initial" state
    frame = 0

    # Number of times we reached the goal
    reached_goal   = 0
    # We use a goal Threshold for the pendulum
    #goal_threshold = -100
    goal_threshold = 1000 #unreachable for the CartPole env
    
    while frame < num_frames and not (reached_goal == 5):
    
        states     = []
        actions    = []
        log_probas = []
        rewards    = []
        values     = []
        not_terminal_states = []

        all_actor_losses  = []
        all_critic_losses = []

        for _ in range(num_steps):
            state = torch.FloatTensor(state).to(device)
        
            # Get the estimate of our policy and our state's value
            #dist, value = model(state)
            dist  = model_actor(state)
            value = model_critic(state)
        
            # Take action probabilistically
            action = dist.sample()

            # Get the associated transition
            next_state, reward, done, info = envs.step(action.cpu().numpy())

            # We are dealing with an environment that does not uses reward at all !
            # Our goals are
            # 1/ to explore (and learn useful things)
            # 2/ to reach goals

            goal_reward = compute_goal_reward(next_state)

            # Compute the "log prob" of our policy
            log_proba = dist.log_prob(action)

            # Update our structures
            states.append(state)

            if discrete_action_space:
                action     = action.unsqueeze(1)
                log_proba  = log_proba.unsqueeze(1)

            
            actions.append(action)
            log_probas.append(log_proba)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            not_terminal_states.append(torch.FloatTensor(1-done).unsqueeze(1).to(device)) # Reward and Value do not have the right shape there
        
            state  = next_state

            frame += 1
        
            if frame % 1000 == 0:
                test_reward = np.mean([test_env(render=False) for _ in range(10)])
                test_rewards.append(test_reward)
                print("Frame %s: got mean reward: %s" % (frame, test_reward))
                if test_reward > goal_threshold:
                    reached_goal += 1
                    print("Frame %s/%s : reached goal for the %s time" % (frame, num_frames, reached_goal))


            #if frame % 5000 == 0:
                #plot(frame, test_rewards)

            if frame % 100 == 0:
                print("Frame %s/%s" % (frame, num_frames))
        

        # Now update
        # First, compute estimated advantages and returns

        next_state = torch.FloatTensor(next_state).to(device)
        #next_dist, next_value = model(next_state)
        next_dist  = model_actor(next_state)
        next_value = model_critic(next_state)
    
        returns = compute_returns(next_value, rewards, not_terminal_states, values)
        
        # Detach the useful tensors
        log_probas = torch.cat(log_probas).detach()
        values     = torch.cat(values).detach()
        returns    = torch.cat(returns).detach()
    
        states     = torch.cat(states)
        actions    = torch.cat(actions)
    
        # Compute the advantages :
        # As returns comes from a GAE, this is supposed
        # to be a 'good' estimation of the advantage
        advantages = returns - values
    
        # Update !
        ppo_full_step(epochs, mini_batch_size, states, actions, log_probas, returns, advantages)

    if plot_final:
        plot(num_frames, test_rewards)

    return num_frames, test_rewards

############################################

if __name__=="__main__":

    training()
    test_env(render=True)
