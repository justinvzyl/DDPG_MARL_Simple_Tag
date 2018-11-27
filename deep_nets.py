import torch.nn as nn

class ActorNet(nn.Module):
    
    def __init__(self, d_in,d_out):
        super(ActorNet,self).__init__()     
        self.action_vector = nn.Sequential(nn.Linear(d_in,32),
                                 nn.ReLU(),
                                 nn.Linear(32,d_out),
                                 nn.Tanh())
              
    def forward(self, x):
        return self.action_vector(x)



class CriticNet(nn.Module):
    
    def __init__(self, d_in,d_out):
        super(CriticNet,self).__init__()     
        self.value = nn.Sequential(nn.Linear(d_in+d_out,32),
                                 nn.ReLU(),
                                 nn.Linear(32,1))
        
    def forward(self, x):
     
        return self.value(x)