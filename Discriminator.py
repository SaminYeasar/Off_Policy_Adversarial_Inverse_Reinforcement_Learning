import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import sys
from torch.distributions import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Policy weights
def weights_init_(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)


"""
reward function and value function both has same network architecture
"""
class CommonNet(nn.Module):
	def __init__(self, in_dim, hid_dim):
		super(CommonNet, self).__init__()
		self.NN = nn.Sequential(
			nn.Linear(in_dim, hid_dim),
			nn.ReLU(),
			nn.Linear(hid_dim, hid_dim),
			nn.ReLU(),
			nn.Linear(hid_dim, 1))
		self.apply(weights_init_)
	def forward(self, x):
		output = self.NN(x)
		return output






class AIRL_func(object):
	def __init__(self, device, args, state_dim, action_dim, hid_dim=None):
		super(AIRL_func, self).__init__()

		self.device = device
		self.args = args
		self.training = True
		self.state_dim = state_dim
		self.action_dim = action_dim


		self.value_func = CommonNet(in_dim=self.state_dim, hid_dim=100).to(self.device)
		self.value_optimizer = torch.optim.Adam(self.value_func.parameters())

		if args.state_only == True:
			# r(s)
			self.reward_func = CommonNet(in_dim=self.state_dim, hid_dim=256).to(self.device)
		else:
			# r(s,a)
			self.reward_func = CommonNet(in_dim=self.state_dim + self.action_dim, hid_dim=256).to(self.device)
		self.reward_optimizer = torch.optim.Adam(self.reward_func.parameters())

	def train(self, mode=True):
		def check_and_set(module):
			assert module.training == self.training
			module.train(mode)
		check_and_set(self.reward_func)
		check_and_set(self.value_func)
		self.training = mode
		return self

	def eval(self):
		return self.train(False)



	def run(self, state, next_state, action, lprobs, critarion=None, generator=None, gamma=0.9):


		# ===========================
		# (1) compute r(s) or r(s,a):
		# ===========================
		""" compare result for both (s) and (s,a) input
			approximating reward only for s makes it "disentangled reward" - AIRL
		"""

		if self.args.state_only == True:
			reward = self.reward_func(state)

		else:
			reward = self.reward_func(torch.cat([state, action], dim=1))




		############################
		# (2) value function shaping
		############################
		# V(s) and V(s')
		# V_s = self.value_func(state)
		# V_ns = self.value_func(next_state)
		V_s = self.value_func(state)
		V_ns = self.value_func(next_state)

		######################################################
		# (3) compute f(s,a,s')
		#
		# Define log p_tau(a|s) = r + gamma * V(s') - V(s)
		######################################################

		# log p_tau(a|s) likelihood of action given state
		# self.qfn = Q(s,a) = r + \gamma * V(s')
		# log p_tau = Q(s,a) - V(s) = A(s,a) = f(s,a,s')

		# computes Q(s,a)
		#Q_value = reward + gamma * V_ns
		# computes f(s,a,s') = log p_tau
		#log_p_tau = reward + gamma * V_ns - V_s

		log_p = reward + gamma * V_ns - V_s
		log_q = lprobs
		log_pq_concat = torch.cat([log_p, log_q], 1)
		log_pq = torch.logsumexp(torch.cat([log_p, log_q], 1).view(len(state), 2), dim=1).view(-1, 1)


		if critarion == 'Expert':
			loss2 = F.binary_cross_entropy_with_logits(log_pq_concat, torch.ones(log_pq_concat.size()).to(self.device), reduction='sum')
			log_D = log_p - log_pq
			D = torch.exp(log_D)
			return D, loss2

		if critarion == 'Policy':
			loss2 = F.binary_cross_entropy_with_logits(log_pq_concat, torch.zeros(log_pq_concat.size()).to(self.device), reduction='sum')
			log_D_ = log_q - log_pq    # log(1-D)
			D = 1 - torch.exp(log_D_)  # exp (log (1-D)) = 1-D; thus D = 1 - (1-D)
			return D, loss2

			




