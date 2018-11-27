from deep_nets import ActorNet, CriticNet
import torch

class Actor():

    def __init__(self, n_o, n_a):
        self.dim_a = n_a
        self.dim_o = n_o
        self.policy = ActorNet(n_o, n_a)

    def get_action(self, state):
        return self.policy(state)


class Critic():
    
    def __init__(self, n_o, n_a):
        self.dim_o = n_o
        self.value_func = CriticNet(n_o, n_a)
        
    
    def get_state_value(self, state, action):
        x = torch.cat((state,action), dim = 1)
        state_value = self.value_func.forward(x)
        return state_value

        