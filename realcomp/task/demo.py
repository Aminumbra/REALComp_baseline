import inspect
import os

import gym
import numpy as np
import realcomp
import tqdm
from realcomp.task import config

print(realcomp)  # this is an hack because disable unused-imports does not work

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
# from my_controller import MyController
from realcomp.task.PPOAgent import PPOAgent, CNN
from MultiprocessEnv import RobotVecEnv, VecNormalize
from train_cnn import train_cnn, test_cnn
import torch.optim as optim
import torch

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


env_id = "REALComp-v0"
envs = [make_env(env_id) for e in range(config.num_envs)]
envs = RobotVecEnv(envs, keys=["joint_positions", "touch_sensors"]) # Add 'retina' and/or 'touch_sensors' if needed
envs = VecNormalize(envs, size_obs_to_norm = 13 + 3*3, ret=True)

loss_function = torch.nn.MSELoss()


#################################################


def demo_run():
    # env = gym.make('REALComp-v0')
    # controller = Controller(env.action_space)
    controller = PPOAgent(action_space=envs.action_space,
                          size_obs=(13 + 3*3)  * config.observations_to_stack, #13 : joints + sensors; 3*3 : 3 coordinates per object, 3 objects
                          shape_pic=None,#(72, 144, 3),  # As received from the wrapper
                          size_layers=[32, 64, 16],
                          size_cnn_output=2,
                          actor_lr=1e-4,
                          critic_lr=1e-3,
                          value_loss_coeff=1.,
                          gamma=0.95,
                          gae_lambda=0.95,
                          epochs=10,
                          horizon=32,
                          mini_batch_size=8,
                          frames_per_action=config.frames_per_action,
                          init_wait=config.noop_steps,
                          clip=0.2,
                          entropy_coeff=0.01,
                          log_std=0.,
                          use_parallel=True,
                          num_parallel=config.num_envs,
                          logs=True,
                          )


    ###################################
    # Pre-training of the CNN

    if config.pre_train_cnn:
        crop = True
        if crop:
            shape_pic = (72, 144, 3)
        else:
            shape_pic = (240, 320, 3)

        model_cnn = CNN(shape_pic=shape_pic, size_output=2)
        model_optimizer = optim.Adam(model_cnn.parameters())
        env = gym.make("REALComp-v0")

        losses = train_cnn(env,
                           model_cnn,
                           model_optimizer,
                           updates=400,
                           shape_pic=shape_pic,
                           crop=crop,
                           tensorboard=config.tensorboard)

        # Quick display of the model performances
        test_cnn(env,
             model_cnn,
             model_optimizer,
             10,
             shape_pic=shape_pic,
             crop=crop)

        # Assign the trained model to the Agent

        controller.cnn = model_cnn.to(config.device)
        controller.optimizer = optim.Adam(params=list(controller.actor.parameters())
                                          + list(controller.critic.parameters()))

    ###################################

    # render simulation on screen
    if config.render:
        envs.render('human')

    # reset simulation
    observation = envs.reset(config.random_reset)
    reward = np.zeros(config.num_envs)
    done = np.zeros(config.num_envs)

    # intrinsic phase
    some_state = np.zeros_like(reward, dtype=np.float64)
    time_since_last_touch = 0
    touches = 0
    new_episode = True

    if config.model_to_load:
        controller.load_models(config.model_to_load)
    else:
        print("Starting intrinsic phase...")
        for frame in tqdm.tqdm(range(config.intrinsic_frames // config.num_envs)):
            # time.sleep(0.05)
            # if config.save_every and frame and frame % config.save_every == 0:
            #     controller.save_models("models.pth")

            # Used to reset the normalization. All the envs. terminate at the same time, so we can do this
            if any(done):
                envs.ret = done
            
            if new_episode:
                new_episode = False
                # Add things : change current goal, etc
                pass

            action = controller.step(observation, reward, done, test=False)

            observation, reward, done, _ = envs.step(action.cpu())
            reward, had_contact, some_state = update_reward(envs, frame, reward, some_state)
            
            time_since_last_touch += 1

            config.tensorboard.add_scalar('intrinsic/rewards', reward.mean(), frame)
            config.tensorboard.add_scalar('intrinsic/did_it_touch', had_contact.max(), frame)
            if had_contact.max():
                config.tensorboard.add_scalar('intrinsic/time_since_last_touch', time_since_last_touch, touches)
                touches += 1
                time_since_last_touch = 0

            if config.pre_train_cnn:
                picture = observation[:, 13:]
                picture = picture.reshape((controller.num_parallel, controller.shape_pic[0], controller.shape_pic[1], controller.shape_pic[2]))
                picture = torch.FloatTensor(picture)
                picture = picture.permute(0, 3, 1, 2)
                cnn_output = controller.cnn(picture.to(config.device))
                cnn_output = cnn_output.detach()
                loss = loss_function(cnn_output.to(torch.device("cpu")), torch.FloatTensor(envs.get_obj_pos("orange")[:, 0:2]).to(torch.device("cpu"))).to(torch.device("cpu")).mean()
                config.tensorboard.add_scalar('train/Fixed_CNN_loss', loss, frame)


            if config.reset_on_touch:
                if any(had_contact) and (frame % config.frames_per_action == 0):
                    done = np.ones(config.num_envs)
                    observation = envs.reset(config.random_reset)
                    new_episode = True

            if (frame > config.noop_steps) and ((frame - config.noop_steps) % (config.frames_per_action * config.actions_per_episode) == 0):
                done = np.ones(config.num_envs)
                observation = envs.reset(config.random_reset)
                new_episode = True
                
    # controller.save_models("models.pth")

    config.tensorboard.close()
    print("Starting extrinsic phase...")
    if config.enjoy:
        input("Press enter to test the agent and visualize its actions !")
        showoff(controller)

    # extrinsic phase
    env = gym.make('REALComp-v0')
    # if config.render:
    env.render('human')

    for k in range(config.extrinsic_trials):

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


def showoff(controller, target="orange", punished_objects=["mustard", "tomato"]):

    envs = [make_env(env_id)]
    envs = RobotVecEnv(envs, keys=["joint_positions", "touch_sensors"]) # Add 'retina' and/or 'touch_sensors' if needed
    envs = VecNormalize(envs, size_obs_to_norm = 13 + 3*3, ret=True)

    envs.render('human')

    controller.soft_reset()
    controller.num_parallel = 1
    observation = envs.reset(config.random_reset)
    reward = None
    done = False
    for frame in tqdm.tqdm(range(10000)):
        action = controller.step(observation, reward, done, test=True)
        observation, reward, done, _ = envs.step(action.cpu())

        good_contacts, bad_contacts = get_contacts(envs, target, punished_objects)

        if (frame > config.noop_steps and frame % 80 == 0) or (good_contacts.max() and config.reset_on_touch):
            observation = envs.reset(config.random_reset)
            done = np.ones(1)


def get_contacts(envs, target_object, punished_objects):
    good_contacts = np.full(len(envs), False)
    bad_contacts  = np.full(len(envs), False)
    envs_contacts = envs.get_contacts()
    robot_useful_parts = ["finger_00", "finger_01", "finger_10", "finger_11"]  # Only care about FINGER contacts

    for i, contacts in enumerate(envs_contacts):
        if contacts:
            for robot_part in contacts:
                if robot_part in robot_useful_parts:  # We are checking if the 'fingers' touched something
                    objects_touched = contacts[robot_part]
                    if any(object_touched == target_object for object_touched in objects_touched):
                        good_contacts[i] = True
                        
                for punished_object in punished_objects: # No contacts AT ALL, not only considering fingers there !
                    objects_touched = contacts[robot_part]
                    if any(object_touched == punished_object for object_touched in objects_touched):
                        bad_contacts[i] = True

    return good_contacts, bad_contacts


def update_reward(envs, frame, reward, some_state, target_object="orange", punished_objects=["mustard", "tomato"]):

    if frame == 0:
        pass

    good_contacts, bad_contacts = get_contacts(envs, target_object, punished_objects)
    robot_useful_parts = ["finger_00", "finger_01", "finger_10", "finger_11"]  # 4 fingers + the last part of the robot (~ "its hand")
    target_pos = envs.get_obj_pos(target_object)

    if frame > config.noop_steps:

        distance_target = np.minimum.reduce([euclidean_distance(target_pos, envs.get_part_pos(robot_part)) for robot_part in robot_useful_parts])

        closeness = np.power(distance_target + 1e-6, -2)
        reward = np.clip(closeness, 0, 10)
        
        # touch_sensors = envs.get_touch_sensors()
        # sensors_activated = np.array(np.max(touch_sensors, axis=1), dtype=bool) # Array of length num_envs, containing True if one of the sensors > 0, False otherwise
        # lift_reward = target_pos[:, 2] * sensors_activated
        
        some_state += reward
        #some_state += lift_reward * 400
        reward = some_state.copy()

        if not frame % config.frames_per_action: # Only interested in the final step of the action for the contact with the target
            reward += 50 * good_contacts - 100 * bad_contacts  # Add an extra-reward for touching the target, and a penalty for touching something else
            some_state.fill(0)

    #assert frame <= config.noop_steps or reward.mean() > 0        
    reward = reward * 0.01

    envs.ret = envs.ret * envs.gamma + reward
    reward = envs._rewfilt(reward)
    
    return reward, good_contacts, some_state


if __name__ == "__main__":
    # os.system('git add .')
    # os.system('git commit -m f"{config.experiment_name}"')
    demo_run()
