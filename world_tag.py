from make_env import make_env
import torch
from DDPG_agents import Agent
from itertools import count

class World():
    def __init__(self, render = False):
        self.env  = make_env('simple_tag')
        self.must_render = render
        
        self.w_state = self.env.reset()
        
        self.n_o_prey = self.env.observation_space[0].shape[0]
        self.n_a_prey = self.env.action_space[0].n
        
        
        
        self.n_o_pred = self.env.observation_space[1].shape[0]
        self.n_a_pred = self.env.action_space[1].n
        
        
        self.predator_agent = Agent('predator',self.n_o_pred, self.n_a_pred)
        self.prey_agent = Agent('prey',self.n_o_prey, self.n_a_prey)
        
        
    def split_data(self, states):
        prey = torch.tensor([states[0]], dtype = torch.float)
        predator = torch.tensor([states[1]], dtype = torch.float)
        return prey, predator
    
            
    def act_eval(self, agent):
        if agent.agent_name == 'predator':
            state = torch.tensor([self.w_state[1]], dtype = torch.float)
        else:
            state = torch.tensor([self.w_state[0]], dtype = torch.float)
            
        action = agent.agent_get_action(state)
        return action

    def train(self):
        
        if self.must_render:
            self.env.render()
        
        prey_action = self.act_eval(self.prey_agent)
        predator_action = self.act_eval(self.predator_agent)
        
        next_state, reward, done, _ = self.env.step([prey_action[0], predator_action[0]])
        
        prey_state, predator_state = self.split_data(next_state)
        
        prey_reward = torch.tensor([reward[0]], dtype = torch.float)
        predator_reward = torch.tensor([reward[1]], dtype = torch.float)
        
        self.predator_agent.agent_train(predator_state, predator_reward)
        self.prey_agent.agent_train(prey_state, prey_reward)
        
        self.w_state = next_state
        if done[0] or done[1]:
            return True
        
        
        

if __name__ == "__main__":
    w = World(render=True)
    
    for i in count():
        d = w.train()
        if d:
            s = w.env.reset()
            w.w_state = s
        if i % 100 == 0 and i is not 0:
            s = w.env.reset()
            w.w_state = s
            