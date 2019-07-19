import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gym
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(PolicyNetwork, self).__init__()
        dim_list = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        for i in range(len(dim_list)-1):
            self.layers.append(nn.Linear(dim_list[i], dim_list[i+1]))
            #self.param_init(self.layers[-1], dim_list[i], dim_list[i+1])
            self.layers.append(nn.ReLU())
        self.out = nn.Linear(dim_list[-1], output_dim)
        #self.param_init(self.out, dim_list[-1], output_dim)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        x = self.sm(x)
        return x

    def param_init(self, layer, dim_in, dim_out):
        with torch.no_grad():
            fan_avg = (dim_in + dim_out) / 2
            limit = np.sqrt(3 / fan_avg)
            layer.weight = nn.Parameter(torch.FloatTensor(dim_out, dim_in).uniform_(-limit, limit))
            if layer.bias is not None:
                layer.bias = nn.Parameter(torch.zeros(dim_out).float())

class Actor(object):
    def __init__(self, input_dim, hidden_dims, output_dim, lr = 1e-4, weight_decay = 1e-6):
        self.model = PolicyNetwork(input_dim, hidden_dims, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if torch.cuda.is_available():
            self.model.to("cuda")

class Reinforce(object):
    "Implements the REINFORCE policy gradient method"
    def __init__(self, actor, env, num_episodes, gamma = 1.0, down_scale = 1, eval_freq = 1, eval_iter = 100):
        self.actor = actor
        self.env = env
        self.dS = self.env.reset().shape[0]
        self.nA = self.env.action_space.n
        self.num_episodes = num_episodes
        self.GPU = torch.cuda.is_available()
        self.gamma = gamma
        self.down_scale = down_scale
        self.eval_freq = eval_freq
        self.eval_iter = eval_iter

    def train(self):
        reward_avgs, reward_stds = [], []
        for e in range(self.num_episodes):
            log_probs, rewards = self.generate_episode()
            rewards /= self.down_scale
            T = len(rewards)
            cum_rewards = np.zeros(T)
            cum_rewards[T-1] = rewards[T-1]
            for i in range(1, T):
                cum_rewards[T-i-1] = rewards[T-i-1] + self.gamma * cum_rewards[T-i]
            self.actor.model.train()
            self.actor.optimizer.zero_grad()
            loss = self.calculate_policy_loss(log_probs, cum_rewards)
            loss.backward()
            self.actor.optimizer.step()
            if e % self.eval_freq == self.eval_freq - 1:
                avg, std = self.evaluate()
                print("Episode {}\taverage reward: {:.3f}\treward std: {:.3f}".format(e+1, avg, std))
                reward_avgs.append(avg)
                reward_stds.append(std)

    def calculate_policy_loss(self, log_probs, cum_rewards):
        cum_rewards = torch.FloatTensor(cum_rewards)
        log_probs = torch.stack(log_probs)
        if self.GPU:
            cum_rewards = cum_rewards.cuda()
        policy_loss = -1.0 * (cum_rewards * log_probs).mean()
        return policy_loss

    def generate_episode(self, render = False):
        self.env.reset()
        self.actor.model.eval()
        log_probs, rewards = [], []
        state = self.env.reset()
        is_terminal = False
        while not is_terminal:
            if render:
                env.render()
            state = torch.Tensor(state.reshape((1, self.dS)))
            if self.GPU: state = state.cuda()
            out = self.actor.model(state)[0]
            m = Categorical(out)
            action = m.sample()
            log_prob = m.log_prob(action)
            state, reward, is_terminal, _ = self.env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
        return log_probs, np.array(rewards)

    def evaluate(self):
        self.actor.model.eval()
        cum_rewards = np.zeros(self.eval_iter)
        for i in range(self.eval_iter):
            state = self.env.reset()
            is_terminal = False
            while not is_terminal:
                state = torch.Tensor(state.reshape((1, self.dS)))
                if self.GPU: state = state.cuda()
                out = self.actor.model(state)[0]
                action = torch.argmax(out).item()
                state, reward, is_terminal, _ = self.env.step(action)
                cum_rewards[i] += reward
        return np.mean(cum_rewards), np.std(cum_rewards)

if __name__ == "__main__":
    actor = Actor(8, [16, 16, 16], 4, lr=5e-4, weight_decay=0.0)
    env = gym.make("LunarLander-v2")
    reinforce = Reinforce(actor, env, num_episodes=30000, down_scale=100, eval_freq=200, eval_iter=50)
    reinforce.generate_episode()
    reinforce.train()
