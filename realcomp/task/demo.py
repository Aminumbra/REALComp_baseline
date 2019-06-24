from OpenGL import GLU
import numpy as np
import realcomp
from realcomp.envs.realcomp_env import Goal
import gym

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
#from my_controller import MyController
from PPOAgent import PPOAgent
from MultiprocessEnv import SubprocVecEnv

Controller = PPOAgent

#################################################
## Vectorizing several environments

def make_env(env_id):
        def _thunk():
            return gym.make(env_id)

        return _thunk

num_envs = 4
env_id = "REALComp-v0"
envs   = [make_env(env_id) for e in range(num_envs)]
envs   = SubprocVecEnv(envs) # Wrapper simulating a threading situation, 1 env/Thread

#################################################



def demo_run(extrinsic_trials=10):

    env = gym.make('REALComp-v0')
    #controller = Controller(env.action_space)
    controller =  Controller(action_space      = env.action_space,
                             size_obs          = 26,
                             size_layers       = [64, 64],
                             actor_lr          = 1e-4,
                             critic_lr         = 1e-3,
                             gamma             = 0.99,
                             gae_lambda        = 0.95,
                             epochs            = 10,
                             horizon           = 64,
                             mini_batch_size   = 16,
                             frames_per_action = 30,
                             init_wait         = 120,
                             clip              = 0.2,
                             entropy_coeff     = 0.05,
                             log_std           = 0.,
                             use_parallel      = True,
                             num_parallel      = num_envs,
                             logs              = False,
                             logs_dir          = "-robot")


    
    
    env.intrinsic_timesteps = 1e5 #2000
    env.extrinsic_timesteps = 100 #10

    # render simulation on screen
    #env.render('human')
    
    
    # reset simulation
    observation = envs.reset()
    
    reward = [0] * num_envs
    done   = [False] * num_envs

    objects_names = ["mustard", "tomato", "orange"]
    
    obj_init_pos = np.ndarray((3, 3))
    obj_cur_pos  = np.ndarray((3, 3))

    frame = 0

    def euclidean_distance(x, y):
        return np.sqrt(sum((x-y)**2))

    # intrinsic phase
    print("Starting intrinsic phase...")
    while not all(done):

        ######################
        # Display information
        
        # if frame % 3000 == 0:
        #     print("Frame (actual actions) : ", frame // controller.frames_per_action, "/", env.intrinsic_timesteps // controller.frames_per_action)
        #     #controller.save_models("models.pth")
            

        #####################
        # Before each 'real' step (i.e. controller.frames_per_action frames)
        # if frame % controller.frames_per_action == 1 and frame >= controller.init_wait:
        #     for i, obj in enumerate(objects_names):
        #         obj_init_pos[i] = env.get_obj_pos(obj)

        #     had_contact = False
                
        # Call your controller to chose action 
        action = controller.step(observation, reward, done)

        frame += 1

        # contacts = env.get_contacts()
        # if contacts:
        #     for robot_part in contacts:
        #         if "finger" in robot_part: # We are checking if the 'fingers' touched something
        #             objects_touched = contacts[robot_part]
        #             if any(object_touched == "orange" for object_touched in objects_touched):
        #                 had_contact = True
        #                 break

        # do action
        observation, reward, done, _ = envs.step(action)


        ######################
        # After each 'real' step
        # if frame % controller.frames_per_action == 0 and frame >= controller.init_wait + 1:
        #     for i, obj in enumerate(objects_names):
        #         obj_cur_pos[i] = env.get_obj_pos(obj)

        #     #reward = 1 if (((obj_init_pos - obj_cur_pos).mean())**2 > 1e-6) else -1   # Reward for moving something. Observations should be images here
        #     #reward = 1 if had_contact else 0   # Reward for touching something
            
        #     distance_orange = min([euclidean_distance(env.get_obj_pos("tomato"), env.get_part_pos("finger_" + digits)) for digits in ["10", "11"]])

        #     reward = 1 - (1. / distance_orange) + 100*had_contact
            
            # if reward > 50:
            #     reward_count += 1
            #     print("High reward ", reward_count, "at frame ", frame // controller.frames_per_action)

        # get frames for video making
        # rgb_array = env.render('rgb_array')

    if controller.logs:
        controller.writer.close()
        
    # extrinsic phase
    print("Starting extrinsic phase...")

    #####TO REMOVE####
    input("Press enter to test the model")

    #######ADDED TO VISUALIZE##############
    env = gym.make('REALComp-v0')
    env.render('human')
    #######################################
        
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
            #action = controller.step(observation, reward, done)

            ##### Modif : action taken WITHOUT "Exploration Noise"
            action = controller.step_opt(observation, reward, done)
            
            # do action
            observation, reward, done, _ = env.step(action)

            # get frames for video making
            # rgb_array = env.render('rgb_array')

if __name__=="__main__":
    demo_run()
