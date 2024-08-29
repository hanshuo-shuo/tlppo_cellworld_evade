from copy import deepcopy
import warnings
import numpy as np
import gym
from envs.wrappers.time_limit import TimeLimit
from envs.wrappers.tensor import TensorWrapper
from envs.exceptions import UnknownTaskError

warnings.filterwarnings('ignore', category=DeprecationWarning)
from envs.prey_env.gym_env_bins import Environment


# def make_env(cfg):
# 	"""
# 	Make an environment for TD-MPC2 experiments.
# 	"""
# 	gym.logger.set_level(40)
# 	if cfg.multitask:
# 		env = make_multitask_env(cfg)
# 	else:
# 		env = None
# 		for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env]:
# 			try:
# 				env = fn(cfg)
# 			except UnknownTaskError:
# 				pass
# 		if env is None:
# 			raise UnknownTaskError(cfg.task)
# 		env = TensorWrapper(env)
# 	try: # Dict
# 		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
# 	except: # Box
# 		cfg.obs_shape = {'state': env.observation_space.shape}
# 	cfg.action_dim = env.action_space.shape[0]
# 	cfg.episode_length = env.max_episode_steps
# 	cfg.seed_steps = max(1000, 5*cfg.episode_length)
# 	return env

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


def make_env(cfg):
	"""
	Make Myosuite environment.
	"""
	env = Environment()
	env = MypreyWrapper(env, cfg)
	env = TimeLimit(env, max_episode_steps=1000)
	env.max_episode_steps = env._max_episode_steps
	return env


def make_prey_env(cfg):
	gym.logger.set_level(40)
	env = make_env(cfg)
	env = TensorWrapper(env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {'state': env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_step
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env
