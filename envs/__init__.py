from copy import deepcopy
import warnings
import numpy as np
import gym
import gymnasium
from envs.wrappers.time_limit import TimeLimit
from envs.wrappers.tensor import TensorWrapper
from envs.exceptions import UnknownTaskError
import cellworld_gym as cwg

warnings.filterwarnings('ignore', category=DeprecationWarning)
from envs.prey_env.gym_env_bins import Environment


class GymnasiumToGymWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        obs, info = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done or truncated, info

    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        self.env.close()

class MypreyWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg

	def step(self, action):
		obs, reward, done, info = self.env.step(action.copy())
		obs = obs.astype(np.float32)
		return obs, reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.render()


def make_env(cfg, max_episode_steps=50):
	"""
	Make Myosuite environment.
	"""
	env = gymnasium.make("CellworldBotEvade-v0",
             world_name="21_05",
             use_lppos=False,
             use_predator=True,
             max_step=300,
             time_step=0.25,
             render=False,
             real_time=False,
             reward_function=cwg.Reward({"puffed": -1, "finished": 1}),
			 action_type=cwg.BotEvadeEnv.ActionType.CONTINUOUS)
	env = GymnasiumToGymWrapper(env)
	# env = Environment()
	env = MypreyWrapper(env, cfg)
	env = TimeLimit(env, max_episode_steps=max_episode_steps)
	env.max_episode_steps = env._max_episode_steps
	return env


def make_prey_env(cfg):
	gym.logger.set_level(40)
	env = make_env(cfg, max_episode_steps=cfg.episode_length)
	env = TensorWrapper(env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {'state': env.observation_space.shape}

	cfg.action_dim = int(env.action_space.shape[0])
	# show the type of action space
	if isinstance(env.action_space, (gym.spaces.Discrete, gymnasium.spaces.Discrete)):
		print('discrete')
	elif isinstance(env.action_space, (gym.spaces.Box, gymnasium.spaces.Box)):
		print('box')
	else:
		raise ValueError('Unknown action space type')
	print(type(cfg.action_dim))
	cfg.episode_length = 100 #env.max_step
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env

if __name__ == '__main__':
	env = gymnasium.make("CellworldBotEvade-v0",
						 world_name="21_05",
						 use_lppos=False,
						 use_predator=True,
						 max_step=300,
						 time_step=0.25,
						 render=False,
						 real_time=False,
						 reward_function=cwg.Reward({"puffed": -1, "finished": 1}))
	env = GymnasiumToGymWrapper(env)
	env.reset()
	env.step(env.action_space.sample())
	print(env.observation_space.shape)
	print(env.action_space.n)