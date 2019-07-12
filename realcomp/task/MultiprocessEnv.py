# This code is from openai baseline
# https://github.com/openai/baselines/tree/master/baselines/common/vec_env
from multiprocessing import Process, Pipe

import numpy as np
import torch

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    position = torch.zeros(env.action_space.shape[0])
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            position += data
            position = torch.clamp(position, -np.pi / 2, np.pi / 2)
            position[-2:] = torch.clamp(position[-2:], 0)
            ob, reward, done, info = env.step(position)
            if done:
                ob = env.reset()
                position = position * 0.
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            if data:
                ob = env.reset(data)
            else:
                ob = env.reset()
            position = position * 0.
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_contacts':
            contacts = env.get_contacts()
            remote.send(contacts)
        elif cmd == 'get_obj_pos':
            obj_pos = env.get_obj_pos(data)
            remote.send(obj_pos)
        elif cmd == 'get_part_pos':
            part_pos = env.get_part_pos(data)
            remote.send(part_pos)
        elif cmd == 'get_touch_sensors':
            touch_sensors = env.robot.get_touch_sensors()
            remote.send(touch_sensors)
        elif cmd == 'render':
            if data is not None:
                env.render(data)
            else:
                env.render()

        else:
            # General thing.
            # CALLS the method, does not return the function : no need to call it later on
            # Not really satisfying, but might be a workaround to use some functions that are
            # not implemented yet.

            attr = getattr(env, cmd)
            if len(data) == 0:
                if callable(attr):
                    remote.send(attr())
                else:
                    remote.send(attr)
            else:
                remote.send(method(data))


class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def render(self, mode=None):
        for remote in self.remotes:
            remote.send(('render', mode))
            break

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def __len__(self):
        return self.nenvs

    def __getattr__(self, attr, *args):
        print("Searching for attribute ", attr)
        for remote in self.remotes:
            remote.send((attr, args))
        return np.stack([remote.recv() for remote in self.remotes])


from PIL import Image

class RobotVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, keys=["joint_positions", "touch_sensors"]):
        super(RobotVecEnv, self).__init__(env_fns)

        self.keys = keys

    def obs_to_array(self, obs):
        converted_obs = []

        #orange_pos = self.get_obj_pos("orange")
        #mustard_pos = self.get_obj_pos("mustard")
        tomato_pos = self.get_obj_pos("tomato")
        
        for i, o in enumerate(obs):
            converted_obs.append(np.concatenate([np.ravel(o[k]) for k in self.keys if k != "retina"]))
            # converted_obs[-1] = np.concatenate((converted_obs[-1], orange_pos[i], mustard_pos[i], tomato_pos[i]))
            converted_obs[-1] = np.concatenate((converted_obs[-1], tomato_pos[i]))

            if "retina" in self.keys:
                image = o["retina"]
                image = image[60:185, 30:285, :]
                image = Image.fromarray(image)
                image = image.resize((144, 72))  # Width, then height
                image = np.ravel(image) / 255.  # Want a 1D-array, of floating-point numbers

                converted_obs[-1] = np.concatenate((converted_obs[-1], image))

        return np.stack(converted_obs)

    def get_contacts(self):
        for remote in self.remotes:
            remote.send(('get_contacts', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_obj_pos(self, obj):
        for remote in self.remotes:
            remote.send(('get_obj_pos', obj))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_part_pos(self, part):
        for remote in self.remotes:
            remote.send(('get_part_pos', part))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_touch_sensors(self):
        for remote in self.remotes:
            remote.send(('get_touch_sensors', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        return self.obs_to_array(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, data=None):
        for remote in self.remotes:
            remote.send(('reset', data))
        return self.obs_to_array([remote.recv() for remote in self.remotes])


class VecNormalize():
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, envs, size_obs_to_norm=13, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.95, epsilon=1e-8, use_tf=False):
        self.envs = envs
        self.size_obs_to_norm = size_obs_to_norm

        if use_tf:
            from baselines.common.running_mean_std import TfRunningMeanStd
            self.ob_rms = TfRunningMeanStd(shape=self.observation_space.shape, scope='ob_rms') if ob else None
            self.ret_rms = TfRunningMeanStd(shape=(), scope='ret_rms') if ret else None

        else:
            from baselines.common.running_mean_std import RunningMeanStd
            self.ob_rms = RunningMeanStd(shape=(size_obs_to_norm,)) if ob else None
            self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.envs.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        #rews = self._rewfilt(rews)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        # Modified this function so it only normalizes the first 13 values of the observation !
        if self.ob_rms:
            trunc_obs = obs[:, :self.size_obs_to_norm]
            self.ob_rms.update(trunc_obs)
            trunc_obs = np.clip((trunc_obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            obs[:, :self.size_obs_to_norm] = trunc_obs            
            return obs
        else:
            return obs

    def _rewfilt(self, rews):
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            return rews
        else:
            return rews

    def reset(self, data=None):
        self.ret = np.zeros(self.num_envs)
        obs = self.envs.reset(data)
        return self._obfilt(obs)

    def __len__(self):
        return len(self.envs)

    def __getattr__(self, attr):
        orig_attr = self.envs.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                # if result == self.wrapped_class:
                   # return self
                return result
            return hooked
        else:
            return orig_attr
