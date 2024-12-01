import os
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')
import copy
import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_prey_env
from tdmpc2 import TDMPC2
import pandas as pd
torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task evaluation)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
	# assert torch.cuda.is_available()
	# assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	# set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.model_size}', 'blue', attrs=['bold']))
	# print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	# if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
	# 	print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
	# 	print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	# Make environment
	env = make_prey_env(cfg)

	# Load agent
	agent = TDMPC2(cfg)
	print(os.getcwd())
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)

	print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	scores = []
	tasks = [cfg.task]
	for task_idx, task in enumerate(tasks):
		task_idx = None
		ep_rewards, ep_successes = [], []
		# Q_var_list, ep_list= [], []
		# obs_list = []
		# action_list = []
		# reward_list = []
		# done_list = []
		# next_obs_list = []
		for i in range(cfg.eval_episodes):
			obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
			if cfg.save_video:
				frames = [env.render()]
			while not done:
				action = agent.act(obs, t0=t==0, task=task_idx)
				# ep_list.append(i+1)
				copied_obs = copy.deepcopy(obs.numpy())
				new_obs, reward, done, info = env.step(action)
				# obs_list.append(copied_obs)
				# action_list.append(action.numpy())
				# # save reward as float, now it is tensor
				# reward_list.append(reward.item())
				# done_list.append(done)
				# next_obs_list.append(copy.deepcopy(new_obs.numpy()))
				obs = new_obs
				# env.render()
				ep_reward += reward
				t += 1
				if cfg.save_video:
					frames.append(env.render())
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if cfg.save_video:
				imageio.mimsave(
					os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
		ep_rewards = np.mean(ep_rewards)
		ep_successes = np.mean(ep_successes)
		# data = {
		# 	"obs": obs_list,
		# 	"action": action_list,
		# 	"reward": reward_list,
		# 	"done": done_list,
		# 	"next_obs": next_obs_list
		# }
		# data = {
		# 	"ep": ep_list,
		# 	"Q_var": Q_var_list
		# }
		# df = pd.DataFrame(data)
		# print(obs_list)
		# df.to_csv("/Users/hanshuo/Documents/project/RL/tlppo_cellworld_evade/tdmpc_var.csv", index=False)
		print(colored(f'  {task:<22}' \
			f'\tR: {ep_rewards:.01f}  ' \
			f'\tS: {ep_successes:.02f}', 'yellow'))



if __name__ == '__main__':
	evaluate()