import numpy as np
import gym
from envs.wrappers.time_limit import TimeLimit
from envs.prey_env.gym_env_bins import Environment

class MypreyWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg

	def step(self, action):
		obs, reward, _, info = self.env.step(action.copy())
		obs = obs.astype(np.float32)
		info['success'] = info['solved']
		return obs, reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.render().copy()


def make_env(cfg):
	"""
	Make Myosuite environment.
	"""
	env = Environment()
	env = MypreyWrapper(env, cfg)
	env = TimeLimit(env, max_episode_steps=300)
	env.max_episode_steps = env._max_episode_steps
	return env
