import torch
from buffer import Memory
import utils
from ACClasses import Actor, Critic
from utils import OU_noise
import torch.nn as nn


class Agent():
    def __init__(self, name, n_o, n_a):
        self.agent_name = name #name of the agent: 'predator' or 'prey'
        self.dtype= torch.float
        self.obs_dim = n_o #observation space dimensions
        self.act_dim = n_a #action space dimensions
        self.GAMMA = 0.99 #discount factor
        self.ALPHA = 1e-3 #learning rate 
        self.BATCH_SIZE = 16 #number of samples in batch
        self.BUFFER_SIZE = 10000
        self.TAU = 0.001
        self.buffer = Memory(self.BUFFER_SIZE)
        self.noise = OU_noise(dt=0.05)
        
        self.online_actor = Actor(self.obs_dim, self.act_dim)
        self.online_critic = Critic(self.obs_dim, 1)
        
        self.target_actor = Actor(self.obs_dim, self.act_dim)
        self.target_critic = Critic(self.obs_dim, 1)
        
        '''Copy Online Parameters to target Parameters'''
        self.target_actor.policy.load_state_dict(self.online_actor.policy.state_dict())
        self.target_critic.value_func.load_state_dict(self.online_critic.value_func.state_dict())
        
        self.optim = torch.optim.Adam(self.online_critic.value_func.parameters(),lr=self.ALPHA)
        
        self.state = None #temp holding of variables before being pushed to replay buffer. 2D tensor [[state]]
        self.act = None #temp holding of variables before being pushed to replay buffer. 2D tensore [[action]]
    
    
    def agent_get_action(self, observation):
        assert type(observation) is torch.Tensor, 'The state is not a tensor! It is of type: %s' % (type(observation))
        self.state = observation 
    
        #get the action using online actor network and add noise for exploration
        action_d = self.online_actor.get_action(self.state) + self.noise.step()
        self.act = action_d.mean().unsqueeze(0).unsqueeze(1)
        #return action to world 
        return action_d.detach().numpy()
    
    def agent_train(self, ns, r, done = False):
        #convert next state and reward to tensors
        #next_state_v = torch.tensor([next_state],dtype=dtype)
        #reward_v = torch.tensor([reward],dtype=dtype)
        
        #save the values in the replay buffer
        self.buffer.push(self.state,self.act, r, ns, done)
        #set the state to the next state to advance agent
        self.state = ns
        
        #if there are enough samples in replay buffer, perform network updates
        if len(self.buffer) >= self.BUFFER_SIZE:
            #get a mini batch from the replay buffer
            sample = self.buffer.sample(self.BATCH_SIZE)
            #make the data nice
            compressed_states, compressed_actions, compressed_next_states, compressed_rewards = utils.extract_data(sample)
            
            #critic network training
            #yt=r(st,at)+γ⋅Q(st+1,μ(st+1))
            na_from_tactor_a = self.target_actor.get_action(compressed_next_states)
            na_from_tactor = na_from_tactor_a.mean(dim=1).unsqueeze(-1)
            v_from_tcritic = self.target_critic.get_state_value(compressed_next_states, na_from_tactor)
            
            #calculate yt=r(st,at)+γ⋅Q(st+1,μ(st+1))
            target_v = compressed_rewards.unsqueeze(1) + self.GAMMA * v_from_tcritic
            actual_v = self.online_critic.get_state_value(compressed_states,compressed_actions)
            loss = nn.MSELoss()
            output = loss(actual_v, target_v)
            self.optim.zero_grad()
            output.backward(retain_graph=True)
            self.optim.step()
            
            self.online_critic.value_func.zero_grad()
            
            for s,a in zip(compressed_states.split(1),compressed_actions.split(1)):
                online_v = self.online_critic.get_state_value(s,a)
                grad_wrt_a = torch.autograd.grad(online_v,(s,a))
                
                action = self.online_actor.get_action(s)
                action.mean().backward(retain_graph=True)
                
                for param in self.online_actor.policy.parameters():
                    param.data += self.ALPHA * (param.grad * grad_wrt_a[1].item())/(self.BATCH_SIZE)
                    
                self.online_actor.policy.zero_grad()
                self.online_critic.value_func.zero_grad()
            
            
    #            #soft update
            
            for param_o, param_t in zip(self.online_actor.policy.parameters(), self.target_actor.policy.parameters()):
                param_t.data = param_o.data * self.TAU + param_t.data * (1 - self.TAU)
                
            
            for param_o, param_t in zip(self.online_critic.value_func.parameters(), self.target_critic.value_func.parameters()):
                param_t.data = param_o.data * self.TAU + param_t.data * (1 - self.TAU)
            
            self.online_actor.policy.zero_grad()
            self.target_actor.policy.zero_grad()
            self.online_critic.value_func.zero_grad()
            self.target_critic.value_func.zero_grad()
            
            torch.save(self.target_actor.policy.state_dict(), self.agent_name + 'target_actor_state_1.pt')
            torch.save(self.target_critic.value_func.state_dict(), self.agent_name + 'target_critic_state_1.pt')
        