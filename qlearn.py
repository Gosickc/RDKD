import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import validate_model
class LossControlEnv:
    def __init__(self, config, vlkd):
        self.config = config
        self.reset()
        self.vlkd = vlkd

    def reset(self):
        self.eta = self.config.eta
        self.beta = self.config.beta
        self.lamb = self.config.lamb
        self.mu = self.config.mu
        self.l1 = self.config.l1
        self.l2 = self.config.l2
        self.l3 = self.config.l3
        self.l4 = self.config.l4
        self.l5 = self.config.l5

        self.state = [self.eta, self.beta, self.lamb, self.mu, self.l1, self.l2, self.l3, self.l4, self.l5]
        return self.state

    def step(self, action):
        # 更新参数
        self.eta += action[0]
        self.beta += action[1]
        self.lamb += action[2]
        self.mu += action[3]
        self.l1 += action[4]
        self.l2 += action[5]
        self.l3 += action[6]
        self.l4 += action[7]
        self.l5 += action[8]

        # 将参数限制在有效范围内
        self.eta = max(0.2, min(0.9, self.eta))
        self.beta = max(0.01, min(0.7, self.beta))
        self.lamb = max(0.01, min(0.8, self.lamb))
        self.mu = max(1, min(2, self.mu))
        self.l1 = max(0.01, min(0.5, self.l1))
        self.l2 = max(0, min(0.5, self.l2))
        self.l3 = max(0.5, min(1.5, self.l3))
        self.l4 = max(0.01, min(0.5, self.l4))
        self.l5 = max(0.01, min(1.1, self.l5))

        # 更新状态
        self.state = [self.eta, self.beta, self.lamb, self.mu, self.l1, self.l2, self.l3, self.l4, self.l5]

        # 计算奖励
        reward = 0
        done = False
        return self.state, reward, done



