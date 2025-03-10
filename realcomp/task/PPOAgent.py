import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from realcomp.task import config
from torch.distributions import Normal


# from matplotlib import pyplot as plt


class ModelActor(nn.Module):

    def __init__(self,
                 size_obs=13,
                 num_actions=9,  # Our robot : 9 angles
                 size_layers=[32, 32],
                 log_std=0.,
                 lr=3e-4):

        super(ModelActor, self).__init__()

        self.layers = nn.ModuleList()
        num_hidden = len(size_layers)

        self.layers.append(nn.Linear(size_obs, size_layers[0]))
        self.layers.append(nn.ReLU())

        for i in range(num_hidden - 1):
            self.layers.append(nn.Linear(size_layers[i], size_layers[i + 1]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(size_layers[num_hidden - 1], num_actions))
                                   
        self.log_std = nn.Parameter(torch.ones(1, num_actions) * log_std)
        self.num_actions = num_actions

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        mu = x  # Might need to rescale it ? tanh ?
        std = self.log_std.exp()

        if mu.dim() > 1:
            std = std.expand_as(mu)

        #mu = torch.tanh(mu) * np.pi / 2
        dist = Normal(mu, std, validate_args=True)
                           
        return dist


class ModelCritic(nn.Module):
    def __init__(self,
                 size_obs=13,
                 size_layers=[32, 32],
                 lr=3e-3):

        super(ModelCritic, self).__init__()

        self.layers = nn.ModuleList()
        num_hidden = len(size_layers)

        self.layers.append(nn.Linear(size_obs, size_layers[0]))
        self.layers.append(nn.ReLU())

        for i in range(num_hidden - 1):
            self.layers.append(nn.Linear(size_layers[i], size_layers[i + 1]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(size_layers[num_hidden - 1], 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class CNN(nn.Module):
    def __init__(self, shape_pic=(96, 144, 3), size_output=256):
        super(CNN, self).__init__()

        self.size_output = size_output

        self.layer1 = nn.Sequential(
            nn.Conv2d(shape_pic[2], 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        size_h = self.size_after_conv(shape_pic[0], 5, 1, 2) #Layer 1 Conv layer
        size_h = self.size_after_conv(size_h, 2, 2, 0) #MaxPool of Layer 1
        size_h = self.size_after_conv(size_h, 5, 1, 2)
        size_h = self.size_after_conv(size_h, 2, 2, 0)
        size_h = self.size_after_conv(size_h, 5, 1, 2)
        size_h = self.size_after_conv(size_h, 2, 2, 0)

        size_w = self.size_after_conv(shape_pic[1], 5, 1, 2)
        size_w = self.size_after_conv(size_w, 2, 2, 0)
        size_w = self.size_after_conv(size_w, 5, 1, 2)
        size_w = self.size_after_conv(size_w, 2, 2, 0)
        size_w = self.size_after_conv(size_w, 5, 1, 2)
        size_w = self.size_after_conv(size_w, 2, 2, 0)

        self.fc = nn.Sequential(
            nn.Linear(size_h * size_w * 16, 256),
            nn.ReLU())

        self.output = nn.Sequential(
            nn.Linear(256, size_output))
            #nn.Linear(256, size_output),
            #nn.ReLU())

        # self.optimizer = optim.Adam(self.parameters())

    def size_after_conv(self, init_size, kernel_size, stride, padding, dilation=1):
        return int((init_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def forward(self, x):
        # img = x[1].numpy()
        # img = np.transpose(img, (1, 2, 0))
        # plt.imshow(img)
        # plt.show()
                           
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.output(x)

        return x


class PPOAgent:

    def __init__(self,
                 action_space,
                 size_obs=13 * config.observations_to_stack,
                 shape_pic=(96, 144, 3),
                 size_goal=0,
                 size_layers=[32, 32],
                 size_cnn_output=256,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 value_loss_coeff=0.1,
                 gamma=0.99,
                 gae_lambda=0.95,
                 epochs=10,
                 horizon=64,
                 mini_batch_size=16,
                 frames_per_action=30,
                 init_wait=300,
                 clip=0.2,
                 entropy_coeff=0.1,
                 log_std=-0.6,
                 use_parallel=False,
                 num_parallel=0,
                 logs=False,
                 logs_dir=""):

        """
        A controller for any continuous task, using PPO algorithm.

        Parameters :

        action_space      : gym.action_space. Action space of a given environment.

        size_obs          : Int. Number of elements per observation.

        shape_pic         : Int triplet. Shape of the (potentially downscaled) input image. The last element is
        the number of channels (3 if RGB, 1 if grayscale). If no picture is given as input, shape_pic should be
        None.

        size_goal         : Int. Number of elements per goal.

        size_layers    : List of int. List of the number of neurons of each hidden layer of the neural network.
        The first layer (input) and the last layer (output) must not be part of this list.

        size_cnn_output   : Int. Size of the output of the CNN, which ends with a fully connected linear layer.

        actor_lr          : Float. Learning rate for the actor network.
        
        critic_lr         : Float. Learning rate for the critic network.

        value_loss_coeff  : Float. Coefficient of the Critic loss compared to the actor loss when backpropagating
        through the common part of the network. The final loss function is :
        Loss = Policy_loss + value_loss_coeff * Value_loss

        gamma             : Float. Discount rate when computing the discounted returns with GAE.

        gae_lambda        : Float. 'Lambda' of the GAE algorithm, used to regulate the bias-variance trade-off.

        epochs            : Int. Number of epochs in each update.

        horizon           : Int. Number of actions taken before each update of the networks.

        mini_batch_size   : Int. Size of each batch when updating the networks.

        frames_per_action : Int. Number of times each action must be repeated. Useful when using environments where
        each action has to be taken for more than one frame, or when computing an action with the policy network is
        really expensive.

        init_wait         : Int. Number of frames spent without taking any action. Useful when using environments
        in which the first frames are irrelevant, or to wait for everything in the environment to stabilize.

        clip              : Float. Clipping parameter for the PPO algorithm.

        entropy_coeff     : Float. Entropy coefficient for the PPO algorithm, used to compute the actor's loss.

        log_std           : Float. Log of the initial standard deviation used to help the exploration of the actor
        network.

        use_parallel      : Bool. If you are using vectorized environments, set this to True, in order to reshape
        all the Tensors accordingly. Otherwise, set it to False.

        num_parallel      : Int. Number of parallel workers. Used to reshape correctly the actions. If use_parallel is
        False, this argument has no effect.

        logs              : Bool. If True, a tensorboardX writer will save relevant informations about the training.

        logs_dir          : Str. Comment appended to the name of the directory of the tensorboardX output.
        """

        self.num_actions = action_space.shape[0]
        self.size_obs = size_obs
        self.shape_pic = shape_pic

        if shape_pic is None:
            size_cnn_output = 0

        self.size_goal = size_goal

        self.first_step = True
        self.device = config.device

        self.use_parallel = use_parallel
        self.num_parallel = num_parallel

        # Hyperparameters

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.value_loss_coeff = value_loss_coeff

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.epochs = epochs
        self.horizon = horizon
        self.mini_batch_size = mini_batch_size

        self.clip = clip
        self.entropy_coeff = entropy_coeff

        self.frames_per_action = frames_per_action

        # Models
        if shape_pic is not None:
            self.cnn = CNN(shape_pic=shape_pic, size_output=size_cnn_output).to(config.device)

        self.actor = ModelActor(self.size_obs + size_cnn_output, self.num_actions, size_layers=size_layers, lr=actor_lr, log_std=log_std).to(config.device)
        self.critic = ModelCritic(self.size_obs + size_cnn_output, size_layers=size_layers, lr=critic_lr).to(config.device)

        self.observations_history = collections.deque(maxlen=config.observations_to_stack)

        # Pseudo-memory to be able to update the policy
        self.frame = 0  # Used to know where we are compared to the horizon
        self.state = None
        self.states = []
        self.actions = []
        self.log_probas = []
        self.rewards = []
        self.values = []
        self.not_done = []

        self.action_to_repeat = None
        self.num_repeated_action = self.frames_per_action  # 'trick' so everything works even at the first step

        self.init_wait = init_wait
        self.already_waited = 0

        # Meta-variable to get some information about the training

        self.number_updates = 0
        self.logs = logs

        if logs:
            # self.writer = UtilsTensorboard.writer(logs_dir)
            self.writer = config.tensorboard

        params = list(self.actor.parameters()) + list(self.critic.parameters())

        if self.shape_pic:
            params += list(self.cnn.parameters())

        self.optimizer = optim.Adam(params=params, lr=config.lr)

    ######################################################################

    # Utility functions
    def convert_observation_to_input(self, observation):
        # retina        = torch.FloatTensor(observation["retina"])
        # retina        = torch.reshape(retina, (-1, 3, 240, 320))
        # goal          = torch.FloatTensor(observation["goal"])
        # goal          = torch.reshape(goal, (-1, 3, 240, 320))

        # x = torch.cat((retina, goal))

        # x = x.reshape(240 * 320 * 2 * 3)

        ###################
        # To work with joints

        if self.use_parallel:
            list_obs = []
            for obs in observation:
                joints = torch.FloatTensor(obs["joint_positions"])
                sensors = torch.FloatTensor(obs["touch_sensors"])

                curr_obs = torch.cat((joints, sensors)).unsqueeze(0)
                list_obs.append(curr_obs)

            x = torch.cat(list_obs)

        else:
            joints = torch.FloatTensor(observation["joint_positions"])
            sensors = torch.FloatTensor(observation["touch_sensors"])
            x = torch.cat((joints, sensors))

        return x

    def compute_reward(self, observation):  # "Observation" is supposed to contain the goal in itself

        retina = torch.FloatTensor(observation["retina"])
        goal = torch.FloatTensor(observation["goal"])

        ###TODO

    def save_models(self, path):

        torch.save({
            "model_actor": self.actor.state_dict(),
            "model_critic": self.critic.state_dict(),
            "optim_actor": self.actor.optimizer.state_dict(),
            "optim_critic": self.critic.optimizer.state_dict()
        }, path)

    def load_models(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["model_actor"])
        self.critic.load_state_dict(checkpoint["model_critic"])
        self.actor.optimizer.load_state_dict(checkpoint["optim_actor"])
        self.critic.optimizer.load_state_dict(checkpoint["optim_critic"])

        # self.actor.eval()
        # self.critic.eval()

    def soft_reset(self):
        self.frame = 0  # Used to know where we are compared to the horizon
        self.state = None
        self.states = []
        self.actions = []
        self.log_probas = []
        self.rewards = []
        self.values = []
        self.not_done = []

        self.action_to_repeat = None
        self.num_repeated_action = self.frames_per_action  # 'trick' so everything works even at the first step

        self.already_waited = 0

        self.first_step = True

    ######################################################################

    # Functions used by the PPO algorithm in itself

    def compute_returns_gae(self, next_value):

        values = self.values + [next_value]  # Can't simply append, as it would modify external values

        advantage = 0
        returns = []

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * self.not_done[step] - values[step]
            advantage = delta + self.gamma * self.gae_lambda * self.not_done[step] * advantage

            returns.insert(0, advantage + values[step])

        return returns

    def ppo_iterator(self, mini_batch_size, states, actions, log_probas, returns, advantages):

        n_states = states.size(0)
        possible_indices = np.arange(n_states)
        
        for k in range(n_states // mini_batch_size):
            # generates mini_batch_size indices
            indices = np.random.choice(possible_indices, mini_batch_size, replace=False)
            possible_indices = np.array([x for x in possible_indices if not x in indices])

            yield (states[indices, :],
                   actions[indices, :],
                   log_probas[indices, :],
                   returns[indices, :],
                   advantages[indices, :])

    def ppo_full_step(self,
                      returns,
                      advantages):

        for k in range(self.epochs):

            for state, action, old_log_probas, return_, advantage in self.ppo_iterator(self.mini_batch_size, self.states, self.actions, self.log_probas, returns, advantages):

                n_states = state.size(0)

                if self.shape_pic is not None:
                    joints, picture = state[:, :self.size_obs], state[:, self.size_obs:]
                    picture = picture.reshape((n_states, self.shape_pic[0], self.shape_pic[1], self.shape_pic[2]))
                    picture = picture.permute(0, 3, 1, 2)
                    cnn_pic = self.cnn(picture)

                    new_state = torch.cat((joints, cnn_pic), 1)
                    dist = self.actor(new_state)
                    value = self.critic(new_state)

                else:
                    dist = self.actor(state)
                    value = self.critic(state)

                entropy = dist.entropy().mean()

                new_log_probas = dist.log_prob(action)

                ratio = (new_log_probas - old_log_probas).exp()  # Not simply new/old, as we have LOG of them

                # Normalize advantage :
                # advantage = (advantage - advantage.mean()) / advantage.std()
                estimate_1 = ratio * advantage
                estimate_2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantage

                # L_CLIP in the paper
                actor_loss = - torch.min(estimate_1, estimate_2).mean()  # We consider the opposite, as we perform gradient ASCENT
                actor_loss -= self.entropy_coeff * entropy

                # L_VF in the paper
                critic_loss = ((return_ - value) ** 2).mean()

                # if self.shape_pic is not None:
                #     self.cnn.optimizer.zero_grad()
                # self.actor.optimizer.zero_grad()
                # self.critic.optimizer.zero_grad()
                self.optimizer.zero_grad()

                loss = actor_loss + self.value_loss_coeff * critic_loss
                loss.backward()
                self.optimizer.step()

            if self.logs:
                self.writer.add_scalar("train/entropy", entropy.mean().item(), self.epochs * self.number_updates + k)
                self.writer.add_scalar("train/actor loss", actor_loss.mean().item(), self.epochs * self.number_updates + k)
                self.writer.add_scalar("train/Critic loss", critic_loss.mean().item(), self.epochs * self.number_updates + k)

                if self.shape_pic is not None:
                    self.writer.add_scalar("train/Shared CNN loss", loss.mean().item(), self.epochs * self.number_updates + k)

    def step(self, observation, reward, done, test=False):  # observation[, :size_obs] is joints+sensors, and observation[, size_obs:] is a flattened picture

        if self.already_waited < self.init_wait:
            self.already_waited += 1
            if self.use_parallel:
                return torch.zeros(self.num_parallel, self.num_actions)
            else:
                return torch.zeros(self.num_actions)

        if self.num_repeated_action < self.frames_per_action:
            self.num_repeated_action += 1
            return self.action_to_repeat.detach()

        if not self.first_step:

            if self.use_parallel:
                self.rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                self.not_done.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device))  # Reward and Value do not have the right shape there

            else:
                self.rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.device))
                self.not_done.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(self.device))  # Reward and Value do not have the right shape there

        self.frame += 1

        if self.frame == self.horizon and not test: # Don't want to learn while testing
            self.update()

        # TODO : Might need to change a few things if we use a CNN on only part of the stacked frames !!
        state_t = observation

        if not self.observations_history:
            for _ in range(config.observations_to_stack - 1):
                self.observations_history.append(state_t)
        self.observations_history.append(state_t)

        if len(self.observations_history) > 1:
            state = torch.cat(tuple(self.observations_history), dim=1)
        else:
            state, = self.observations_history

        state = torch.FloatTensor(state).to(self.device)
        self.state = state

        # Get the estimate of our policy and our state's value

        # Split into 2 different part
        if self.shape_pic is not None:

            joints, picture = state[:, :self.size_obs], state[:, self.size_obs:]
            picture = picture.reshape((self.num_parallel, self.shape_pic[0], self.shape_pic[1], self.shape_pic[2]))
            # picture = np.transpose(picture, (0, 3, 1, 2))
            picture = picture.permute(0, 3, 1, 2)
            cnn_pic = self.cnn(picture)

            new_state = torch.cat((joints, cnn_pic), 1)
            dist = self.actor(new_state)
            value = self.critic(new_state)

        else:
            dist = self.actor(state)
            value = self.critic(state)

        # Take action probabilistically
        if test:
            dist.scale = torch.full_like(dist.scale, 0.2) # Use a very low std, to visualize what the "real" policy is

        action = dist.sample()

        # Compute the "log prob" of our policy
        log_proba = dist.log_prob(action)

        # Update our structures

        self.actions.append(action)
        self.log_probas.append(log_proba)

        if self.use_parallel:
            self.states.append(state)
            self.values.append(value)

        else:
            self.states.append(state.unsqueeze(0))
            self.values.append(value.unsqueeze(1))

        self.first_step = False

        if not self.use_parallel:
            self.action_to_repeat = action.reshape(self.num_actions)
        else:
            self.action_to_repeat = action

        # Reset the repeat-action counter
        self.num_repeated_action = 1

        action_detached = self.action_to_repeat.detach()
        return action_detached

    def update(self):
        """
        Considers that we have made 'horizon' steps, and we got the
        associated rewards/states/goals/etc.
        Computes the returns, and updates the model.
        """

        # Now update
        # First, compute estimated advantages and returns

        next_state = self.state

        if self.shape_pic is not None:
            joints, picture = next_state[:, :self.size_obs], next_state[:, self.size_obs:]
            picture = picture.reshape((self.num_parallel, self.shape_pic[0], self.shape_pic[1], self.shape_pic[2]))
            picture = picture.permute(0, 3, 1, 2)
            cnn_pic = self.cnn(picture)

            new_state = torch.cat((joints, cnn_pic), 1)
            next_dist = self.actor(new_state)
            next_value = self.critic(new_state)

        else:
            next_dist = self.actor(next_state)
            next_value = self.critic(next_state)

        returns = self.compute_returns_gae(next_value)

        # Detach the useful tensors
        self.log_probas = torch.cat(self.log_probas).detach()
        self.values = torch.cat(self.values).detach()

        returns = torch.cat(returns).detach()

        self.states = torch.cat(self.states)
        self.actions = torch.cat(self.actions)

        # Compute the advantages :
        # As returns comes from a GAE, this is supposed
        # to be a 'good' estimation of the advantage
        advantages = returns - self.values

        # Update !
        self.ppo_full_step(returns, advantages)

        if self.logs:
            self.writer.add_scalar("train/Rewards", torch.cat(self.rewards).mean().item(), self.number_updates)
            self.writer.add_scalar("train/Values", self.values.mean().item(), self.number_updates)
            self.writer.add_scalar("train/Log std", self.actor.log_std.mean().item(), self.number_updates)

        # Reset the attributes
        self.states = []
        self.actions = []
        self.log_probas = []
        self.rewards = []
        self.values = []
        self.not_done = []

        self.frame = 0

        self.number_updates += 1

    #############

    def step_opt(self, observation, reward, done):

        self.num_repeated_action += 1

        if self.num_repeated_action < self.frames_per_action:
            return self.action_to_repeat.detach()

        self.state = self.convert_observation_to_input(observation)

        state = torch.FloatTensor(self.state).to(self.device)

        # Get the estimate of our policy and our state's value
        dist = self.actor(state)

        action = dist.mean

        self.action_to_repeat = action.reshape(self.num_actions)
        # Reset the repeat-action counter
        self.num_repeated_action = 0

        action_detached = self.action_to_repeat.detach()
        return action_detached
