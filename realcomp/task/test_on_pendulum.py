import gym
import torch
from PPOAgent import PPOAgent
import sys, signal
from MultiprocessEnv import SubprocVecEnv

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

    return env, envs, size_obs, num_actions, discrete_action_space, num_envs

# Run the function now, as several things are global variables
# used later in the code

env, envs, size_obs, num_actions, discrete_action_space, num_envs = create_envs(env_id="Pendulum-v0", num_envs=8)

############################################

def convert_observation_to_input(obs):
    return torch.FloatTensor(obs)


controller = Controller(action_space      = env.action_space,
                        size_obs          = size_obs,
                        size_layers       = [64, 64],
                        actor_lr          = 1e-4,
                        critic_lr         = 1e-3,
                        gamma             = 0.99,
                        gae_lambda        = 0.95,
                        epochs            = 10,
                        horizon           = 64,
                        mini_batch_size   = 16,
                        frames_per_action = 1,
                        init_wait         = 0,
                        clip              = 0.2,
                        entropy_coeff     = 0.005,
                        log_std           = 0.,
                        use_parallel      = True,
                        logs              = False,
                        logs_dir          = "-pendulum")


controller.convert_observation_to_input = convert_observation_to_input

signal.signal(signal.SIGINT, signal.default_int_handler)


state  = envs.reset()
reward = 0
done   = 0

frame = 0

try:
    while frame < 20000:
        frame += 1
    
        action = controller.step(state, reward, done)
        
        state, reward, done, _ = envs.step(action)
        
        if frame % 5000 == 0:
            print("Frame ", frame)
        
            
except KeyboardInterrupt:
    if controller.logs:
        controller.writer.close()

    print("Exiting properly")

    sys.exit()


#######################################  
