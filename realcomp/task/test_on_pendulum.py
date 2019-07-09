import gym
import torch
from PPOAgent import PPOAgent
import sys, signal
from MultiprocessEnv import SubprocVecEnv
import tqdm

Controller = PPOAgent


######### TEST ON PENDULUM ############

def make_env(env_id):
        def _thunk():
            return gym.make(env_id)

        return _thunk
    
def create_envs(env_id="Pendulum-v0", num_envs=16):
    ##currently working environment : Pendulum-v0, BipedalWalker-v2
    #Continuous action space, continous observation space
    
    env = gym.make(env_id) # current, testing env

    envs = [make_env(env_id) for e in range(num_envs)]
    envs = SubprocVecEnv(envs) # Wrapper simulating a threading situation, 1 env/Thread

    size_obs = envs.observation_space.shape[0]

    num_actions = envs.action_space.shape[0]

    print("Env name : ", env_id)
    print("Different elements per observation : ", size_obs)
    print("Number of actions : ", num_actions)
    print("Simulated environments : ", num_envs)

    return env, envs, size_obs, num_actions, num_envs

# Run the function now, as several things are global variables
# used later in the code

env, envs, size_obs, num_actions, num_envs = create_envs(env_id="Pendulum-v0", num_envs=8)

############################################

def convert_observation_to_input(obs):
    return torch.FloatTensor(obs)


controller = PPOAgent(action_space=envs.action_space,
                          size_obs=size_obs,
                          shape_pic=None,
                          size_layers=[256],
                          size_cnn_output=2,
                          actor_lr=1e-3,
                          critic_lr=1e-3,
                          value_loss_coeff=1.,
                          gamma=0.99,
                          gae_lambda=0.95,
                          epochs=4,
                          horizon=32,
                          mini_batch_size=8,
                          frames_per_action=1,
                          init_wait=1,
                          clip=0.2,
                          entropy_coeff=0.01,
                          log_std=0.,
                          use_parallel=True,
                          num_parallel=8,
                          logs=True,
                          )


signal.signal(signal.SIGINT, signal.default_int_handler)


state  = envs.reset()
reward = 0
done   = 0

def showoff(envs, controller):
    observation = envs.reset()
    reward = None
    done = [False]

    controller.soft_reset()
    controller.num_parallel = 1
        
    while not done[0]:
        envs.render()
        action = controller.step(observation, reward, done, test=True)
        observation, reward, done, _ = envs.step(action.cpu())

        
try:
    for frame in tqdm.tqdm(range(200000)):
            
        action = controller.step(state, reward, done, test=False)
        
        state, reward, done, _ = envs.step(action.cpu())

    input("Press enter to visualize the agent !")

    envs = [make_env("Pendulum-v0")]
    envs = SubprocVecEnv(envs)

    for _ in range(5):
        showoff(envs, controller)
            
except KeyboardInterrupt:
        
    input("Press enter to visualize the agent !")

    envs = [make_env("Pendulum-v0")]
    envs = SubprocVecEnv(envs)

    for _ in range(5):
        showoff(envs, controller)
        
    print("Exiting properly")

    sys.exit()


#######################################  
