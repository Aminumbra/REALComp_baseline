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
from realcomp.task.PPOAgent import PPOAgent
from PPOAgent import PPOAgent
from MultiprocessEnv import VecNormalize

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
envs = VecNormalize(envs, keys=["joint_positions", "touch_sensors", "retina"], ret=False)  # Add 'retina' if needed


#################################################


def demo_run():
    # env = gym.make('REALComp-v0')
    # controller = Controller(env.action_space)
    controller = PPOAgent(action_space=envs.action_space,
                          size_obs=13 * config.observations_to_stack,
                          shape_pic=(96, 144, 3),  # As received from the wrapper
                          size_layers=[64, 64],
                          size_cnn_output=256,
                          actor_lr=1e-4,
                          critic_lr=1e-3,
                          value_loss_coeff=1.,
                          gamma=0.9,
                          gae_lambda=0.95,
                          epochs=10,
                          horizon=64,
                          mini_batch_size=16,
                          frames_per_action=config.frames_per_action,
                          init_wait=config.noop_steps,
                          clip=0.2,
                          entropy_coeff=0.01,
                          log_std=0.,
                          use_parallel=True,
                          num_parallel=config.num_envs,
                          logs=True,
                          )

    # render simulation on screen
    if config.render:
        envs.render('human')

    # reset simulation
    observation = envs.reset()
    reward = np.zeros(config.num_envs)
    done = np.zeros(config.num_envs)

    # intrinsic phase
    some_state = np.zeros_like(reward, dtype=np.float64)

    if config.model_to_load:
        controller.load_models(config.model_to_load)
    else:
        print("Starting intrinsic phase...")
        for frame in tqdm.tqdm(range(config.intrinsic_frames // config.num_envs)):
            # time.sleep(0.05)
            # if config.save_every and frame and frame % config.save_every == 0:
            #     controller.save_models("models.pth")

            action = controller.step(observation, reward, done, test=False)

            observation, reward, done, _ = envs.step(action.cpu())
            reward, had_contact, some_state = update_reward(envs, frame, reward, some_state)

            config.tensorboard.add_scalar('intrinsic/rewards', reward.mean(), frame)
            config.tensorboard.add_scalar('intrinsic/did_it_touch', had_contact.max(), frame)

            if config.reset_on_touch:
                if any(had_contact):
                    done = np.ones(config.num_envs)
                    observation = envs.reset()

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


def showoff(controller):
    envs = [make_env(env_id)]
    envs = VecNormalize(envs, keys=["joint_positions", "touch_sensors"])  # Add 'retina' if needed

    envs.render('human')

    controller.soft_reset()
    controller.num_parallel = 1
    observation = envs.reset()
    reward = None
    done = False
    for frame in tqdm.tqdm(range(10000)):
        action = controller.step(observation, reward, done, test=True)
        observation, reward, done, _ = envs.step(action.cpu())

        if get_contacts(envs, "orange")[0]:
            observation = envs.reset()
            done = np.ones(1)


def get_contacts(envs, target_object):
    had_contact = np.full(len(envs), False)
    envs_contacts = envs.get_contacts()
    robot_useful_parts = ["finger_00", "finger_01", "finger_10", "finger_11"]  # Only care about FINGER contacts

    for i, contacts in enumerate(envs_contacts):
        if contacts:
            for robot_part in contacts:
                if robot_part in robot_useful_parts:  # We are checking if the 'fingers' touched something
                    objects_touched = contacts[robot_part]
                    if any(object_touched == target_object for object_touched in objects_touched):
                        had_contact[i] = True
                        break

    return had_contact


def update_reward(envs, frame, reward, some_state, goal=None, target_object="orange"):
    obj_init_pos = np.ndarray((3, len(envs), 3))
    obj_cur_pos = np.ndarray((3, len(envs), 3))

    if frame == 0:
        pass

    elif frame % config.frames_per_action == 0 and frame >= config.noop_steps:
        for i, obj in enumerate(objects_names):
            obj_init_pos[i] = envs.get_obj_pos(obj)  # We associate to the i-th object an array of length len(envs),  whose elements are positions (3-tuples)

    had_contact = get_contacts(envs, target_object)
    robot_useful_parts = ["finger_00", "finger_01", "finger_10", "finger_11", "lbr_iiwa_link_7"]  # 4 fingers + the last part of the robot (~ "its hand")

    if frame > config.noop_steps:
        for i, obj in enumerate(objects_names):
            obj_cur_pos[i] = envs.get_obj_pos(obj)

        distance_target = np.minimum.reduce([euclidean_distance(envs.get_obj_pos(target_object), envs.get_part_pos(robot_part)) for robot_part in robot_useful_parts])

        closeness = np.power(distance_target + 1e-6, -2)
        reward = np.clip(closeness, 0, 10)

        some_state += reward
        reward = some_state.copy()

        if not frame % config.frames_per_action:
            reward += 100 * had_contact  # Add an extra-reward for touching the orange
            some_state.fill(0)

    assert frame <= config.noop_steps or reward.mean() > 0
    reward = reward * 0.01
    return reward, had_contact, some_state


if __name__ == "__main__":
    # os.system('git add .')
    # os.system('git commit -m f"{config.experiment_name}"')
    demo_run()
