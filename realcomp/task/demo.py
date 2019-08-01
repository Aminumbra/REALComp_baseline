import inspect
import os
import time

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
from scipy.interpolate import interp1d

no_operation = torch.zeros((config.num_envs, 9))
objects_names = ["mustard", "tomato", "orange", "cube"]

target = "cube"
punished_objects = []

max_diff = np.stack([[0.06, 0.06, 0.05, 0.06, 0.05, 0.1, 0.1, 0.1, 0.1] for _ in range(config.num_envs)])

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
envs = VecNormalize(envs, size_obs_to_norm = 13 + 3*1 + 3*1, ret=True)

cnn_loss_function = torch.nn.MSELoss()


#################################################


def demo_run():
    # env = gym.make('REALComp-v0')
    # controller = Controller(env.action_space)
    controller = PPOAgent(action_space=envs.action_space,
                          size_obs=(13 + 3*1 + 3*1) * config.observations_to_stack, #13 : joints + sensors; 3*1 : 3 coordinates per object, 1 object here
                          shape_pic=None,#(72, 144, 3),  # As received from the wrapper
                          size_layers=[256, 128],
                          size_cnn_output=2,
                          actor_lr=1e-4,
                          critic_lr=1e-3,
                          value_loss_coeff=1.,
                          gamma=0.95,
                          gae_lambda=0.95,
                          epochs=5,
                          horizon=16,
                          mini_batch_size=4,
                          frames_per_action=config.frames_per_action,
                          init_wait=config.noop_steps,
                          clip=0.2,
                          entropy_coeff=0.01,
                          log_std=0.3, # TODO : see if this is an actual impact. Default : 0.0
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
    acc_reward = np.zeros_like(reward, dtype=np.float64)
    time_since_last_touch = 0
    touches = 0
    new_episode = True

    init_position = envs.get_obj_pos(target)
    # If we want a fixed goal : (center = desired_goal, radius=0)
    new_goals = gen_random_goals(center=init_position, radius=np.ones(config.num_envs) * 0.15)
    envs.set_goal_position(new_goals)

    if config.model_to_load:
        controller.load_models(config.model_to_load)

    print("Starting intrinsic phase...")
    for frame in tqdm.tqdm(range(config.intrinsic_frames // config.num_envs)):
        # time.sleep(0.05)
        # if config.save_every and frame and frame % config.save_every == 0:
        #     controller.save_models("models.pth")

        # Used to reset the normalization. All the envs. terminate at the same time, so we can do this
        if any(done):
            envs.ret *= done

        if new_episode:
            new_episode = False
            # Add things : change current goal, etc
            pass

        action = controller.step(observation, reward, done, test=False).cpu().numpy()
        action_fast = action[:, :controller.num_actions]
        action_slow = action[:, controller.num_actions:]

        for _ in range(25):
            envs.step(action_fast)

        for _ in range(125):
            current_joints = envs.get_joint_positions()
            desired_joints = action_slow
            desired_joints = np.clip(desired_joints, -np.pi/2, np.pi/2) # Don't want to interpolate between 'wrong' values !

            current_action = limitActionByJoint(current_joints, desired_joints, max_diff)
            observation, reward, done, _ = envs.step(current_action)

        reward, had_contact, acc_reward = update_reward(envs,
                                                        frame,
                                                        reward,
                                                        acc_reward,
                                                        target=target,
                                                        punished_objects=punished_objects,
                                                        action=action,
                                                        init_position=init_position,
                                                        goal_position=envs.goal_position)

        time_since_last_touch += 1

        #if frame % 10 == 0:
            #config.tensorboard.add_scalar('Rewards/frame_rewards', reward.mean(), frame)

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
            loss = cnn_loss_function(cnn_output.to(torch.device("cpu")), torch.FloatTensor(envs.get_obj_pos(target)[:, 0:2]).to(torch.device("cpu"))).to(torch.device("cpu")).mean()
            config.tensorboard.add_scalar('train/Fixed_CNN_loss', loss, frame)


        if (frame > config.noop_steps) and ((frame + 1 - config.noop_steps) % config.frames_per_action == 0): # True on the last frame of an action

            config.tensorboard.add_scalar('intrinsic/actions_magnitude', abs(action).mean(), frame / config.frames_per_action)

            if config.reset_on_touch and any(had_contact):
                done = np.ones(config.num_envs)
                target_pos = envs.get_obj_pos(target)
    
                if config.random_goal == "random":
                    new_goals = gen_random_goals(center=target_pos, radius=np.ones(config.num_envs) * 0.15)
                    envs.set_goal_position(new_goals)

                if any(euclidean_distance(init_position, target_pos) > 0.03):
                    observation = envs.reset(config.random_reset)
                    for _ in range(15):
                        envs.step(no_operation)
                        
                else:
                    for _ in range(30):
                        observation, _, _, _ = envs.step(no_operation)
                init_position = envs.get_obj_pos(target)
                new_episode = True

        if (frame > config.noop_steps) and ((frame + 1 - config.noop_steps) % (config.frames_per_action * config.actions_per_episode) == 0):
            done = np.ones(config.num_envs)
            target_pos = envs.get_obj_pos(target)
            
            if config.random_goal == "random":
                new_goals = gen_random_goals(center=target_pos, radius=np.ones(config.num_envs) * 0.15)
                envs.set_goal_position(new_goals)
                
            if any(euclidean_distance(init_position, target_pos) > 0.03):
                observation = envs.reset(config.random_reset)
                for _ in range(15):
                    envs.step(no_operation)
                    
            else:
                for _ in range(30):
                    observation, _, _, _ = envs.step(np.zeros((config.num_envs, 9)))
            init_position = envs.get_obj_pos(target)
            new_episode = True

    config.tensorboard.close()
    controller.save_models("models/" + config.experiment_name + ".pth")
    
    print("Starting extrinsic phase...")
    if config.enjoy:
        input("Press enter to test the agent and visualize its actions !")
        showoff(controller, target=target, punished_objects=punished_objects)

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
    envs = VecNormalize(envs, size_obs_to_norm = 13 + 3*1 + 3*1, ret=True)

    init_position = envs.get_obj_pos(target)
    # If we want a fixed goal : (center = desired_goal, radius=0)
    new_goals = gen_random_goals(center=init_position, radius=np.ones(1) * 0.15)
    envs.set_goal_position(new_goals)
    
    envs.render('human')

    controller.soft_reset()
    controller.num_parallel = 1
    observation = envs.reset(config.random_reset)
    reward = None
    done = False
    num_episodes = 1
    
    for frame in tqdm.tqdm(range(20 * config.frames_per_action * config.actions_per_episode)):
        action = controller.step(observation, reward, done, test=True).cpu().numpy()
        action_fast = action[:, :controller.num_actions]
        action_slow = action[:, controller.num_actions:]
    
        for _ in range(25):
            envs.step(action_fast)

        for _ in range(125):
            current_joints = envs.get_joint_positions()
            desired_joints = action_slow
            desired_joints = np.clip(desired_joints, -np.pi/2, np.pi/2) # Don't want to interpolate between 'wrong' values !
        
            current_action = limitActionByJoint(current_joints, desired_joints, max_diff)
            observation, reward, done, _ = envs.step(current_action)

        #sensors = envs.get_touch_sensors()
        #if any(sensors[0]):
            # print(sensors[0])
            # print(envs.get_contacts())

        if (frame > config.noop_steps) and ((frame + 1 - config.noop_steps) % (config.frames_per_action * config.actions_per_episode) == 0):
            done = np.ones(config.num_envs)

            if num_episodes % 5 == 0:
                done = np.ones(config.num_envs)
                target_pos = envs.get_obj_pos(target)
    
                if config.random_goal == "random":
                    new_goal = gen_random_goals(center=target_pos, radius=np.ones(1) * 0.15)
                    envs.set_goal_position(new_goal)
                    print("Goal changed : now trying to achieve goal ", new_goal)

                if any(euclidean_distance(init_position, target_pos) > 0.03):
                    observation = envs.reset(config.random_reset)
                    for _ in range(15):
                        envs.step(no_operation)
                else:
                    for _ in range(30):
                        envs.step(np.zeros((config.num_envs, 9)))
                
            observation = envs.reset(config.random_reset)
            num_episodes += 1

        # if (frame > config.noop_steps) and ((frame + 1 - config.noop_steps) % config.frames_per_action == 0): # True on the last frame of an action
        #     good_contacts, bad_contacts = get_contacts(envs, target, punished_objects=[], robot_useful_parts=["base"])
        #     if config.reset_on_touch and any(good_contacts):
        #         done = np.ones(config.num_envs)
        #         observation = envs.reset(config.random_reset)
        #         num_episodes += 1

            
def get_contacts(envs, target, punished_objects, robot_useful_parts=["finger_10", "finger_11"]):
    good_contacts = np.full(len(envs), False)
    bad_contacts  = np.full(len(envs), False)
    envs_contacts = envs.get_contacts()

    for i, contacts in enumerate(envs_contacts):
        if contacts:
            for robot_part in contacts:
                objects_touched = contacts[robot_part]
                if robot_part in robot_useful_parts:  # We are checking if the 'fingers' touched something
                    if target in objects_touched:
                        good_contacts[i] = True
                        
                for punished_object in punished_objects: # No contacts AT ALL, not only considering fingers there !
                    if punished_object in objects_touched:
                        bad_contacts[i] = True

    return good_contacts, bad_contacts


def update_reward(envs, frame, reward, acc_reward, init_position, goal_position, target="orange", punished_objects=["mustard", "tomato"], action=0):

    if frame == 0:
        pass

    robot_useful_parts = ["base"]  # 4 fingers + the last part of the robot (~ "its hand")
    good_contacts, bad_contacts = get_contacts(envs, target, punished_objects, robot_useful_parts=robot_useful_parts)#["skin_00", "skin_01", "skin_10", "skin_11"])
    
    target_pos = envs.get_obj_pos(target)

    if frame > config.noop_steps:

        distance_robot_target = np.minimum.reduce([euclidean_distance(target_pos, envs.get_part_pos(robot_part)) for robot_part in robot_useful_parts])
        distance_target_goal  = euclidean_distance(target_pos[:, :2], goal_position[:, :2])
        init_distance = euclidean_distance(init_position[:, :2], goal_position[:, :2])
        
        closeness = np.power(distance_robot_target + 1e-6, -2)
        closeness_reward = np.clip(closeness, 0, 100)

        goal_closeness = np.power(distance_target_goal + 1e-6, -2)
        #goal_closeness = np.clip(goal_closeness, 0, 100)
        
        init_closeness = np.power(init_distance + 1e-6, -2)
        #init_closeness = np.clip(init_closeness, 0, 100) # Useful when the target is randomly respawned near the goal

        goal_closeness_reward = 5 * (goal_closeness - init_closeness)

        moved_object = euclidean_distance(target_pos, init_position) > 0.03
        bonus_moving_object = 2 * moved_object
        
        action_magnitude_penalty = 0 #abs(action).mean(1) # Avoids shaky movements
        bad_contacts_penalty = 0 #30 * bad_contacts  # The penalty for touching something else is always active, not only on last frame
        #reward = closeness_reward + 5 * good_contacts + goal_closeness_reward - action_magnitude_penalty - bad_contacts_penalty
        reward = goal_closeness_reward + bonus_moving_object

        if (frame + 1 - config.noop_steps) % config.frames_per_action == 0:
            config.tensorboard.add_scalar("Rewards/Reward_distance_hand_target", closeness_reward.mean(), frame)
            config.tensorboard.add_scalar("Rewards/Reward_distance_target_goal", goal_closeness_reward.mean(), frame)
            config.tensorboard.add_scalar("Rewards/Distance_improvement", (init_distance - distance_target_goal).mean(), frame)
            #config.tensorboard.add_scalar("Rewards/Good_contacts", good_contacts.mean(), frame)
            #config.tensorboard.add_scalar("Rewards/Bad_contacts", bad_contacts.mean(), frame)

        #if not frame % config.frames_per_action:
            #acc_reward.fill(0)

    reward = reward * 0.01
    acc_reward += reward
    
    if (frame > config.noop_steps and ((frame + 1 - config.noop_steps) % (config.frames_per_action * config.actions_per_episode) == 0)): # True at the last frame of an EPISODE
        # Uncomment this line to have reward_episode = last_reward_of_episode
        # Comment to have reward_episode = sum(reward for reward in rewards_episode)
        acc_reward = reward
        config.tensorboard.add_scalar('Rewards/episode_rewards', acc_reward.mean(), frame / (config.frames_per_action * config.actions_per_episode))
        envs.ret = envs.ret * envs.gamma + acc_reward
        acc_reward = envs._rewfilt(acc_reward)

        return acc_reward, good_contacts, np.zeros_like(acc_reward)

    else:
        return np.zeros_like(acc_reward), good_contacts, acc_reward


    
def limitActionByJoint(current_joints, desired_joints, max_diff):
    min_diff = -max_diff
    diff = np.clip(desired_joints - current_joints, -max_diff, max_diff)
    return current_joints + diff


def gen_random_goals(center=[[0., 0.]], radius = [[1.]]):

    goals = np.zeros((config.num_envs, 3))
    a = np.random.random(config.num_envs) * 2 * np.pi
    r = radius * np.sqrt(np.random.random(config.num_envs))

    x = np.cos(a) * r
    y = np.sin(a) * r
    
    for i in range(config.num_envs):
        init_x = center[i][0]
        init_y = center[i][1]
        goals[i] = np.array([init_x + x[i], init_y + y[i], 0.41])

    return goals

if __name__ == "__main__":
    # os.system('git add .')
    # os.system('git commit -m f"{config.experiment_name}"')
    demo_run()
