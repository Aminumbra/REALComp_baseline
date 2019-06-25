import inspect
import tqdm
import os

import config
import gym
import numpy as np
import realcomp
from realcomp.envs.realcomp_env import Goal
import gym
import pybullet

print(realcomp)  # this is an hack because disable unused-imports does not work

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
# from my_controller import MyController
from PPOAgent import PPOAgent
from MultiprocessEnv import SubprocVecEnv, RobotVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

Controller = PPOAgent

objects_names = ["mustard", "tomato", "orange"]


def euclidean_distance(x, y):
    if len(x.shape) <= 1:
        axis = 0
    else:
        axis = 1
        
    return np.sqrt(np.sum((x - y) ** 2, axis=axis))


#################################################
## Vectorizing several environments

def make_env(env_id):
        def _thunk():
            return gym.make(env_id) 

        return _thunk

num_envs = 4
env_id = "REALComp-v0"
envs   = [make_env(env_id) for e in range(num_envs)]
envs   = RobotVecEnv(envs) # Wrapper simulating a threading situation, 1 env/Thread

#################################################



def demo_run(extrinsic_trials=10):

    #env = gym.make('REALComp-v0')
    #controller = Controller(env.action_space)
    controller =  Controller(action_space      = envs.action_space,
                             size_obs          = 13,
                             size_goal         = 0,
                             size_layers       = [64, 64],
                             actor_lr          = 1e-4,
                             critic_lr         = 1e-3,
                             gamma             = 0.99,
                             gae_lambda        = 0.95,
                             epochs            = 10,
                             horizon           = 64,
                             mini_batch_size   = 16,
                             frames_per_action = config.frames_per_action,
                             init_wait         = config.noop_steps,
                             clip              = 0.2,
                             entropy_coeff     = 0.05,
                             log_std           = 0.,
                             use_parallel      = True,
                             num_parallel      = num_envs,
                             logs              = False,
                             logs_dir          = "-robot")

    #env.intrinsic_timesteps = 1e5  # 2000
    #env.extrinsic_timesteps = 100  # 10

    # render simulation on screen
    if config.render:
        env.render('human')

    # reset simulation
    observation = envs.reset()
    reward = [0] * num_envs
    done   = [False] * num_envs

    # objects_names = ["mustard", "tomato", "orange"]
    
    # obj_init_pos = np.ndarray((3, 3))
    # obj_cur_pos  = np.ndarray((3, 3))

    frame = 0

    # intrinsic phase
    print("Starting intrinsic phase...")
    for frame in tqdm.tqdm(range(config.intrinsic_frames)):
        if all(done):
            break

        if frame and frame % 3000 == 0:
            controller.save_models("models.pth")

        action = controller.step(observation, reward, done)
        observation, reward, done, _ = envs.step(action.cpu())

        frame += 1

        reward = update_reward(envs, frame, reward)

    if controller.logs:
        controller.writer.close()

        

    # extrinsic phase
    print("Starting extrinsic phase...")

    input("Press enter to test the model")

    env = gym.make('REALComp-v0')
    if config.render:
        env.render('human')

    for k in range(extrinsic_trials):

        # reset simulation
        observation = env.reset()
        reward = 0
        done = False

        # set the extrinsic goal to pursue 
        env.set_goal()
        print("Starting extrinsic trial...")

        while not done:
            # Call your controller to chose action
            # action = controller.step(observation, reward, done)

            ##### Modif : action taken WITHOUT "Exploration Noise"
            action = controller.step_opt(observation, reward, done)

            # do action
            observation, reward, done, _ = env.step(action)

            # get frames for video making
            # rgb_array = env.render('rgb_array')



            

def update_reward(envs, frame, reward, goal=None):
    obj_init_pos = np.ndarray((3, len(envs), 3))
    obj_cur_pos = np.ndarray((3, len(envs), 3))

    objects = ["mustard", "orange", "tomato"]

    if frame == 0:
        pass
    
    elif frame % config.frames_per_action == 0 and frame >= config.noop_steps:
        for i, obj in enumerate(objects_names):
            obj_init_pos[i] = envs.get_obj_pos(obj) # We associate to the i-th object an array of length len(envs),  whose elements are positions (3-tuples)
                                                    
    had_contact = np.full(len(envs), False)
    envs_contacts = envs.get_contacts()

    for i, contacts in enumerate(envs_contacts):
        if contacts:
            for robot_part in contacts:
                if "finger" in robot_part:  # We are checking if the 'fingers' touched something
                    objects_touched = contacts[robot_part]
                    if any(object_touched == "orange" for object_touched in objects_touched):
                    #if any(object_touched == objects[goal] for object_touched in objects_touched): #Assumed 'goal' to be an integer ranging from 0 to 2
                        had_contact[i] = True
                        break

    if frame % config.frames_per_action == 0 and frame >= config.noop_steps + 1:
        for i, obj in enumerate(objects_names):
            obj_cur_pos[i] = envs.get_obj_pos(obj)

        # reward = 1 if (((obj_init_pos - obj_cur_pos).mean())**2 > 1e-6) else -1   # Reward for moving something. Observations should be images here
        # reward = had_contact   # Reward for touching something

        distance_orange = euclidean_distance(envs.get_obj_pos("orange"), envs.get_part_pos("finger_10"))

        reward = 1 - (1. / (distance_orange + 1)) + 100 * had_contact  # Avoids division by Zero. Add an extra-reward for touching the orange

    return reward


if __name__ == "__main__":
    demo_run()
