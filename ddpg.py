import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from envs.env_ import ArmEnv
import time
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.layer1(state))
        a = F.relu(self.layer2(a))
        return self.max_action * torch.tanh(self.layer3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400 + action_dim, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, state, act):
        q = F.relu(self.layer1(state))
        q = F.relu(self.layer2(torch.cat((q, act), 1)))
        return self.layer3(q)


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.9, tau=0.0005, memory_capacity=1000):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.002)

        self.action_dim, self.state_dim, self.a_bound = action_dim, state_dim, max_action
        self.discount = discount
        self.tau = tau
        self.memory_capacity = memory_capacity
        self.replay_buffer = []
        self.pointer = 0

    def select_action(self, state):
        # 将state整理为1行的矩阵，生成FloatTensor格式变量
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # 将action转为np.array格式，降为1维向量输出
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=32, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        # Sample replay buffer
        ind = np.random.randint(0, len(self.replay_buffer), size=batch_size)
        x, y, u, r, d = [], [], [], [], []
        for i in ind:
            i, j, kp, kd, X, Y, U, R, D = self.replay_buffer[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(1-np.array(D, copy=False))

        state = torch.FloatTensor(x).to(device)
        action = torch.FloatTensor(u).to(device)
        next_state = torch.FloatTensor(y).to(device)
        not_done = torch.FloatTensor(d).to(device)
        reward = torch.FloatTensor(r).to(device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, i, j, kp, kd, s, s_, a, r, d):
        transition = [i, j, kp, kd, s, s_, a, [r], [d]]
        # i: episode number, j: step number
        if len(self.replay_buffer) == self.memory_capacity:
            index = self.pointer % self.memory_capacity  # replace the old memory with new memory
            self.replay_buffer[index] = transition
        else:
            self.replay_buffer.append(transition)
        self.pointer += 1

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))

    def load(self, filename, directory):
        actor_path = glob.glob('%s/%s_actor.pth' % (directory, filename))[0]
        self.actor.load_state_dict(torch.load(actor_path))
        actor_optimizer_path = glob.glob('%s/%s_actor_optimizer.pth' % (directory, filename))[0]
        self.actor_optimizer.load_state_dict(torch.load(actor_optimizer_path))
        critic_path = glob.glob('%s/%s_critic.pth' % (directory, filename))[0]
        self.critic.load_state_dict(torch.load(critic_path))
        critic_optimizer_path = glob.glob('%s/%s_critic_optimizer.pth' % (directory, filename))[0]
        self.critic_optimizer.load_state_dict(torch.load(critic_optimizer_path))
        print('actor path: {}, critic path: {}'.format(actor_path, critic_path))