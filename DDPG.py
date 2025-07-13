import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ActorNetwork, CriticNetwork
from global_memory import experience_pool
import random
import logging
import numpy as np


class DDPG:
    def __init__(self, state_dim, action_dim):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.actor = ActorNetwork(state_dim, action_dim).cuda()
        self.critic = CriticNetwork(state_dim, action_dim).cuda()
        self.target_actor = ActorNetwork(state_dim, action_dim).cuda()
        self.target_critic = CriticNetwork(state_dim, action_dim).cuda()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.batch_size = 1
        self.gamma = 0.95
        self.tau = 0.01

    def select_action(self, state, epoch):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        action = self.actor(state,epoch).detach().cpu().numpy()[0]
        return action

    def update(self, experience_pool, epoch):

        if len(experience_pool) < self.batch_size:
            return


        batch = random.sample(experience_pool, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(np.array(states)).cuda()
        actions = torch.FloatTensor(np.array(actions)).cuda()
        rewards = torch.FloatTensor(rewards).unsqueeze(1).cuda()
        next_states = torch.FloatTensor(np.array(next_states)).cuda()


        with torch.no_grad():
            next_actions = self.target_actor(next_states, epoch)
            target_q_values = rewards + self.gamma * self.target_critic(next_states, next_actions)
        q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states, epoch)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.logger.info("DDPG update completed: Critic loss = {:.4f}, Actor loss = {:.4f}".format(
            critic_loss.item(), actor_loss.item()))
