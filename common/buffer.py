from pathlib import Path
import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.envs import RandomCropTensorDict, Transform, Compose

from common.logger import make_dir


class DataPrepTransform(Transform):
	"""
	Preprocesses data for TD-MPC2 training.
	Replay data is expected to be a TensorDict with the following keys:
		obs: observations
		action: actions
		reward: rewards
		task: task IDs (optional)
	A TensorDict with T time steps has T+1 observations and T actions and rewards.
	The first actions and rewards in each TensorDict are dummies and should be ignored.
	"""

	def __init__(self):
		super().__init__([])
	
	def forward(self, td):
		td = td.permute(1,0)
		return td['obs'], td['action'][1:], td['reward'][1:].unsqueeze(-1), (td['task'][0] if 'task' in td.keys() else None)


class Buffer():
	"""
	Create a replay buffer for TD-MPC2 training.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self._capacity = min(cfg.buffer_size, cfg.steps)//cfg.episode_length
		self._num_eps = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity
	
	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		Uses the RandomSampler to sample trajectories,
		and the RandomCropTensorDict transform to crop trajectories to the desired length.
		DataPrepTransform is used to preprocess data to the expected format in TD-MPC2 updates.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=RandomSampler(),
			pin_memory=True,
			prefetch=1,
			transform=Compose(
				RandomCropTensorDict(self.cfg.horizon+1, -1),
				DataPrepTransform(),
			),
			batch_size=self.cfg.batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		# mem_free, _ = torch.cuda.mem_get_info()
		mem_free = 0
		bytes_per_ep = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for k,v in tds.items()
		])		
		print(f'Bytes per episode: {bytes_per_ep:,}')
		total_bytes = bytes_per_ep*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		if 2.5*total_bytes > mem_free: # Insufficient CUDA memory
			print('Using CPU memory for storage.')
			return self._reserve_buffer(
				LazyTensorStorage(self._capacity, device=torch.device('cpu'))
			)
		else: # Sufficient CUDA memory
			print('Using CUDA memory for storage.')
			return self._reserve_buffer(
				LazyTensorStorage(self._capacity, device=torch.device('cuda'))
			)

	def add(self, tds):
		"""Add an episode to the buffer. All episodes are expected to have the same length."""
		if self._num_eps == 0:
			self._buffer = self._init(tds)
		self._buffer.add(tds)
		self._num_eps += 1
		return self._num_eps

	def sample(self):
		"""Sample a batch of sub-trajectories from the buffer."""
		obs, action, reward, task = self._buffer.sample(batch_size=self.cfg.batch_size)
		return obs.to(self._device, non_blocking=True), \
			   action.to(self._device, non_blocking=True), \
			   reward.to(self._device, non_blocking=True), \
			   task.to(self._device, non_blocking=True) if task is not None else None

	# def sample(self):
	# 	"""
    #     Sample a batch of sub-trajectories from the buffer with balanced reward types:
    #     - 30% negative reward experiences
    #     - 30% positive reward experiences
    #     - 40% neutral reward experiences
    #     """
	# 	batch_size = self.cfg.batch_size
	#
	# 	# Sample a larger pool to ensure we have enough of each type
	# 	obs, action, reward, task = self._buffer.sample(batch_size=batch_size * 10)
	#
	# 	# Identify different reward types across the horizon
	# 	is_negative = (reward < 0).any(dim=0)
	# 	is_positive = (reward > 0).any(dim=0)
	# 	is_neutral = (~is_negative) & (~is_positive)
	#
	# 	neg_indices = torch.where(is_negative)[0]
	# 	pos_indices = torch.where(is_positive)[0]
	# 	neutral_indices = torch.where(is_neutral)[0]
	#
	# 	# Calculate desired counts for each type
	# 	neg_count = int(batch_size * 0.3)
	# 	pos_count = int(batch_size * 0.2)
	# 	neutral_count = batch_size - neg_count - pos_count
	#
	# 	# Helper function to sample indices with fallback
	# 	def sample_indices(source_indices, desired_count, fallback_indices):
	# 		if len(source_indices) >= desired_count:
	# 			return source_indices[torch.randperm(len(source_indices))[:desired_count]]
	# 		else:
	# 			remaining = desired_count - len(source_indices)
	# 			additional = fallback_indices[torch.randperm(len(fallback_indices))[:remaining]]
	# 			return torch.cat([source_indices, additional])
	#
	# 	# Sample balanced indices
	# 	selected_neg = sample_indices(neg_indices, neg_count, neutral_indices)
	# 	selected_pos = sample_indices(pos_indices, pos_count, neutral_indices)
	#
	# 	# Remaining neutral samples
	# 	remaining_neutral_count = batch_size - len(selected_neg) - len(selected_pos)
	# 	selected_neutral = neutral_indices[torch.randperm(len(neutral_indices))[:remaining_neutral_count]]
	#
	# 	# Combine all selected indices
	# 	selected_indices = torch.cat([selected_neg, selected_pos, selected_neutral])
	# 	selected_indices = selected_indices[torch.randperm(len(selected_indices))]
	#
	# 	# Select the final batch
	# 	obs = obs[:, selected_indices]
	# 	action = action[:, selected_indices]
	# 	reward = reward[:, selected_indices]
	# 	task = task[selected_indices] if task is not None else None
	#
	# 	return obs.to(self._device, non_blocking=True), \
	# 		action.to(self._device, non_blocking=True), \
	# 		reward.to(self._device, non_blocking=True), \
	# 		task.to(self._device, non_blocking=True) if task is not None else None

	def save(self):
		"""Save the buffer to disk. Useful for storing offline datasets."""
		td = self._buffer._storage._storage.cpu()
		fp = make_dir(Path(self.cfg.buffer_dir) / self.cfg.task / str(self.cfg.seed)) / f'{self._num_eps}.pt'
		torch.save(td, fp)
