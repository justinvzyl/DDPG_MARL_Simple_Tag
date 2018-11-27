import torch
import random
import math
from torch.functional import F

def extract_data(sample):
    states_tuple = tuple([_.state for _ in sample])
    actions_tuple = tuple([_.action for _ in sample])
    next_states_tuple = tuple([_.next_state for _ in sample])
    rewards_tuple = tuple([_.reward for _ in sample])

    compressed_rewards = torch.cat(rewards_tuple,dim=0)
    compressed_states = torch.cat(states_tuple,dim=0).requires_grad_()
    compressed_actions = torch.cat(actions_tuple,dim=0).requires_grad_()
    compressed_next_states = torch.cat(next_states_tuple,dim=0)
    
    
    return  F.normalize(compressed_states),F.normalize(compressed_actions), F.normalize(compressed_next_states), compressed_rewards


class OU_noise():
    def __init__(self, th = 1, mu = 0, sig = 1, dt = 1):
        self.theta = th
        self.mu = mu
        self.sigma = sig
        self.dt = dt
        self.x = 0
    
    def reset(self):
        self.t = 0
    
    def step(self):
        x_1 = self.x + self.theta * (self.mu - self.x) * self.dt + self.sigma * math.sqrt(self.dt) * random.normalvariate(0,1)
        self.x = x_1
        return self.x