import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.distributions import Normal
from torch.distributions import multivariate_normal
from torch.distributions import Normal, Categorical

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-8

# From https://github.com/openai/spinningup/blob/master/spinup/algos/sac/core.py


def gaussian_likelihood(noise, log_std):
    pre_sum = -0.5 * noise.pow(2) - log_std
    return pre_sum.sum(-1, keepdim=True) - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def apply_squashing_func(mu, pi, log_pi):
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action, K_primitives):
#         super(Actor, self).__init__()
#
#         self.l1 = nn.Linear(state_dim, 256)
#         self.l2 = nn.Linear(256, 256)
#         self.l3 = nn.Linear(256, 2 * action_dim)
#
#         self.apply(weight_init)
#
#     def forward(self, x, compute_pi=True, compute_log_pi=True):
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         mu, log_std = self.l3(x).chunk(2, dim=-1)
#
#         log_std = torch.tanh(log_std)
#         log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
#             log_std + 1)
#
#         if compute_pi:
#             std = log_std.exp()
#             noise = torch.randn_like(mu)
#             pi = mu + noise * std
#         else:
#             pi = None
#
#         if compute_log_pi:
#             log_pi = gaussian_likelihood(noise, log_std)
#         else:
#             log_pi = None
#
#         mu, pi, log_pi = apply_squashing_func(mu, pi, log_pi)
#
#         return mu, pi, log_pi


class gating_func(nn.Module):
    def __init__(self, num_inputs, hidden_dim=256, K_primitives=8):
        super(gating_func, self).__init__()
        self.NN_w = nn.Sequential(nn.Linear(num_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, K_primitives),
            nn.Sigmoid())

        self.apply(weight_init)

    def forward(self, state):
        return self.NN_w(state)



class sample_gating_func(nn.Module):
    def __init__(self, num_inputs, hidden_dim=256, K_primitives=8):
        super(sample_gating_func, self).__init__()
        self.NN_w = nn.Sequential(nn.Linear(num_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2*K_primitives))

        self.apply(weight_init)

    def scale(self, log_std):
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1)
        return log_std


    def forward(self, state, training=False):
        mean, log_std = self.NN_w(state).chunk(2, dim=-1)
        log_std = self.scale(log_std)
        std = torch.max(log_std.exp(), torch.tensor(1e-6).to(device)) #min std=1e-6
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        pi = torch.tanh(x_t)
        # log_prob = normal.log_prob(x_t)
        # # Enforcing Action Bound
        # log_prob -= torch.log(1 - pi.pow(2) + EPS)
        # log_prob = log_prob.sum(1, keepdim=True)

        if training:
            return pi

        else:
            return mean.cpu().data.numpy().flatten()



class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(num_inputs, 512)
        self.linear2 = nn.Linear(512, 256)


        self.l1 = nn.Sequential( nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 2 * num_actions))
        self.l2 = nn.Sequential( nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 2 * num_actions))
        self.l3 = nn.Sequential( nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 2 * num_actions))
        self.l4 = nn.Sequential( nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 2 * num_actions))
        self.l5 = nn.Sequential( nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 2 * num_actions))
        self.l6 = nn.Sequential( nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 2 * num_actions))
        self.l7 = nn.Sequential( nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 2 * num_actions))
        self.l8 = nn.Sequential( nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 2 * num_actions))

        self.f = nn.Softmax()

        # self.mu_l1 = nn.Linear(hidden_dim, num_actions)
        # self.log_std_l1 = nn.Linear(hidden_dim, num_actions)
        #
        # self.mu_l2 = nn.Linear(hidden_dim, num_actions)
        # self.log_std_l2 = nn.Linear(hidden_dim, num_actions)
        #
        # self.mu_l3 = nn.Linear(hidden_dim, num_actions)
        # self.log_std_l3 = nn.Linear(hidden_dim, num_actions)
        #
        # self.mu_l4 = nn.Linear(hidden_dim, num_actions)
        # self.log_std_l4 = nn.Linear(hidden_dim, num_actions)

        # self.NN_w = nn.Sequential(nn.Linear(num_inputs, hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, K_primitives))


        self.apply(weight_init)

    def scale(self, log_std):
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1)
        return log_std



    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean_1, log_std_1 = self.l1(x).chunk(2, dim=-1)
        mean_2, log_std_2 = self.l2(x).chunk(2, dim=-1)
        mean_3, log_std_3 = self.l3(x).chunk(2, dim=-1)
        mean_4, log_std_4 = self.l4(x).chunk(2, dim=-1)
        mean_5, log_std_5 = self.l5(x).chunk(2, dim=-1)
        mean_6, log_std_6 = self.l6(x).chunk(2, dim=-1)
        mean_7, log_std_7 = self.l7(x).chunk(2, dim=-1)
        mean_8, log_std_8 = self.l8(x).chunk(2, dim=-1)



        # mean_1 = F.relu(self.mu_l1(x))
        # log_std_1 = self.scale(F.relu(self.log_std_l1(x)))
        #
        # mean_2 = F.relu(self.mu_l2(x))
        # log_std_2 = self.scale(F.relu(self.log_std_l2(x)))
        #
        # mean_3 = F.relu(self.mu_l3(x))
        # log_std_3 = self.scale(F.relu(self.log_std_l3(x)))
        #
        # mean_4 = F.relu(self.mu_l4(x))
        # log_std_4 = self.scale(F.relu(self.log_std_l4(x)))

        log_std_list = torch.stack(( self.scale(log_std_1), self.scale(log_std_2), self.scale(log_std_3), self.scale(log_std_4),
                                     self.scale(log_std_5), self.scale(log_std_6), self.scale(log_std_7), self.scale(log_std_8)), 1)
        mean_list = torch.stack((mean_1, mean_2, mean_3, mean_4,
                                 mean_5, mean_6, mean_7, mean_8), 1)


        return mean_list, log_std_list

        # # W = self.NN_w(x)
        # # Computing W parameters:
        # W = gating_func(x)
        #
        # log_std_list = [log_std_1, log_std_2, log_std_3, log_std_4]
        # mean_list = [mean_1, mean_2, mean_3, mean_4]
        #
        # # MCP: Compute equation3
        # mean, std = self.mul_comp_policy(mean_list, log_std_list, W)
        #
        #
        # if compute_pi and compute_log_pi:
        #     normal = MultivariateNormal(mean, std)
        #     x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        #     pi = torch.tanh(x_t)
        #     log_prob = normal.log_prob(x_t)
        #     # Enforcing Action Bound
        #     log_prob -= torch.log(1 - pi.pow(2) + EPS)
        #     log_prob = log_prob.sum(1, keepdim=True)
        # else:
        #     log_prob = None
        #     pi = None
        # return torch.tanh(mean), pi, log_prob






class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        self.apply(weight_init)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, args, K_primitives=8):
        self.training = True
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # separate gating network:
        self.gating_func = gating_func(state_dim, hidden_dim=256, K_primitives=8).to(device)
        self.gating_optimizer = torch.optim.Adam(self.gating_func.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.log_alpha = torch.tensor(np.log(args.initial_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha])

        self.max_action = max_action
        self.K_primitives = K_primitives
        self.action_dim = action_dim
        self.args = args

    def train(self, mode=True):
        def check_and_set(module):
            assert module.training == self.training
            module.train(mode)

        check_and_set(self.actor)
        check_and_set(self.critic)
        check_and_set(self.critic_target)

        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def set_lr(self, lr):
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = lr

        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr

        for param_group in self.log_alpha_optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def alpha(self):
        return self.log_alpha.exp()


    def mul_comp_policy(self, mean_list, log_std_list, W_list):


        m = W_list.unsqueeze(2).expand(mean_list.shape[0], self.K_primitives, self.action_dim)/log_std_list.exp()
        mean = (m*mean_list).sum(dim=1, keepdim=True)/m.sum(dim=1, keepdim=True)
        std = 1/m.sum(dim=1, keepdim=True)

        return mean.squeeze(1), std.squeeze(1)


    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)

            # MCP:
            W_list = self.gating_func(state)
            mean_list, log_std_list = self.actor(state)
            mean, std = self.mul_comp_policy(mean_list, log_std_list, W_list)
            mean = torch.tanh(mean)

        return mean.cpu().data.numpy().flatten()

    def sample_action(self, state, training=False):
        if training == False:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # MCP:
        W_list = self.gating_func(state)
        mean_list, log_std_list = self.actor(state)
        mean, std = self.mul_comp_policy(mean_list, log_std_list, W_list)

        std = torch.max(std, torch.tensor(1e-6).to(device)) #min std=1e-6
        normal = Normal(mean, std)
        #normal = multivariate_normal.MultivariateNormal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        pi = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - pi.pow(2) + EPS)
        log_prob = log_prob.sum(1, keepdim=True)

        if training:
            return mean, pi, log_prob

        else:
            return mean.cpu().data.numpy().flatten(), \
                   pi.cpu().data.numpy().flatten(), log_prob.cpu().data.numpy().flatten()



    def compute_pdf(self, state, action):
        with torch.no_grad():
            # MCP:
            W_list = self.gating_func(state)
            mean_list, log_std_list = self.actor(state)
            mean, std = self.mul_comp_policy(mean_list, log_std_list, W_list)

            #std = log_std.exp()
            dist = Normal(mean, std)
            lprob = dist.log_prob(action).sum(-1, keepdim=True)

            # N, d = mu.shape
            # cov = log_std.exp().pow(2)
            # diff = action - mu
            # exp_term = -0.5 * (diff.pow(2)/cov).sum(dim=1,keepdim=True)
            # norm_term = -0.5 * torch.tensor(2*math.pi).log()*d
            # var_term = -0.5 * cov.log().sum(dim=1, keepdim=True)
            # lprob = norm_term + var_term + exp_term

        if torch.isnan(torch.log(std)).any() or torch.isnan(mean).any() == True:
            print("WARNING: policy : compute_pdf")

        #_,_, lprobs = apply_squashing_func(mu, action, lprob)

        if torch.isnan(lprob).any() == True:
            print("WARNING: policy : compute_pdf")

        return lprob


    def run(self, replay_buffer, num_iterations, tracker, batch_size=100,
                discount=0.99, tau=0.005, policy_freq=2, discriminator=None, predict_reward=None, target_entropy=None):

        for it in range(num_iterations):
            # Sample replay buffer
            # ( state, next_state, action, reward, lprob, done )
            state, next_state, action, lprobs, reward, done = replay_buffer.sample(batch_size)

            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(1 - done).to(device)
            reward = torch.FloatTensor(reward).to(device)
            lprobs = torch.FloatTensor(lprobs).to(device)

            if torch.isnan(state).any() or torch.isnan(action).any() == True:
                print("WARNING")
            # lprobs = self.compute_pdf(state, action).to(device)

            if torch.isnan(lprobs).any() == True:
                print("WARNING")

            """working: just state input
                NOTE: MUST not create any graph from discriminator
            """
            # predicted_reward = discriminator.reward_func(torch.cat([state, action], dim=1))
            # predicted_reward = predicted_reward.detach()
            predicted_reward = predict_reward(state, action)

            # _ , D, _ = discriminator.run(state, next_state, action,
            #                                     torch.ones(state.size()).to(device), lprobs)
            # predicted_reward = discriminator.run(state, next_state, action, lprobs, critarion = 'Reward').detach()

            # input label doesn't matter here as it only requires to compute loss
            # and we just need D here.
            """ try this: D = torch.sigmoid(D).detach()"""
            # D = torch.sigmoid(D).detach()
            # D = torch.exp(log_D).detach()
            # predicted_reward = D.log() - (1-D).log()
            # predicted_reward = log_ptau.detach()

            tracker.update('train_reward', reward.sum().item(), reward.size(0))
            tracker.update('train_predicted_reward', predicted_reward.sum().item(), predicted_reward.size(0))

        # tracker.update('reward_pearsonr', scipy.stats.pearsonr(
        #     reward.cpu().numpy(), predicted_reward.cpu().numpy())[0][0])

            def fit_critic():
                with torch.no_grad():
                    #_, policy_action, log_pi = self.actor(next_state)
                    _, policy_action, log_pi = self.sample_action(next_state, training=True)
                    target_Q1, target_Q2 = self.critic_target(
                        next_state, policy_action)
                    target_V = torch.min(target_Q1,
                                         target_Q2) - self.alpha.detach() * log_pi
                    target_Q = predicted_reward + (done * discount * target_V)

                # Get current Q estimates
                current_Q1, current_Q2 = self.critic(state, action)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                    current_Q2, target_Q)
                tracker.update('critic_loss', critic_loss.detach() * target_Q.size(0), target_Q.size(0))

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            fit_critic()

            def fit_actor():
                # Compute actor loss
                #_, pi, log_pi = self.actor(state)
                _, pi, log_pi = self.sample_action(state,training=True)
                actor_Q1, actor_Q2 = self.critic(state, pi)

                actor_Q = torch.min(actor_Q1, actor_Q2)

                actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
                tracker.update('actor_loss', actor_loss.detach() * state.size(0), state.size(0))

                """Optimize the actor and gating function"""
                if self.args.learn_actor: self.actor_optimizer.zero_grad()
                self.gating_optimizer.zero_grad()

                actor_loss.backward()

                self.gating_optimizer.step()
                if self.args.learn_actor: self.actor_optimizer.step()

                if target_entropy is not None:
                    self.log_alpha_optimizer.zero_grad()
                    alpha_loss = (
                        self.alpha * (-log_pi - target_entropy).detach()).mean()
                    alpha_loss.backward()
                    self.log_alpha_optimizer.step()

            if it % policy_freq == 0:
                fit_actor()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(),
                                               self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data +
                                            (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(),
                   '%s/actor.pth' % (filename))
        torch.save(self.critic.state_dict(),
                   '%s/critic.pth' % (filename))
        torch.save(self.gating_func.state_dict(),
                   '%s/gating.pth' % (filename))

    def load(self, filename):
        self.actor.load_state_dict(
            torch.load('%s/actor.pth' % (filename)))
        self.critic.load_state_dict(
            torch.load('%s/critic.pth' % (filename)))
        self.gating_func.load_state_dict(
            torch.load('%s/gating.pth' % (filename)))