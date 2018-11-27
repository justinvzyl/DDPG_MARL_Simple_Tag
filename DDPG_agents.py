import torch
import gym
from buffer import Memory
import utils
from collections import deque
from ACClasses import Actor, Critic
from utils import OU_noise
import torch.nn as nn
from itertools import count
from tensorboardX import SummaryWriter

dtype= torch.float
GAMMA = 0.99 #discount factor
ALPHA = 1e-3 #learning rate 
BATCH_SIZE = 16 #number of samples in batch
REWARD_THRESHOLD = 200
BUFFER_SIZE = 10000
TAU = 0.001
Total_Reward_List = deque(maxlen=100)
mean_100 = 0
episode_num = 0
buffer = Memory(BUFFER_SIZE)
env_name = 'Pendulum-v0'
env = gym.make(env_name)
noise = OU_noise(dt=0.05)
writer = SummaryWriter(log_dir='./test1')

MAX_STEPS = 200


online_actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
online_critic = Critic(env.observation_space.shape[0], env.action_space.shape[0])

target_actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
target_critic = Critic(env.observation_space.shape[0], env.action_space.shape[0])

'''Copy Online Parameters to target Parameters'''
target_actor.policy.load_state_dict(online_actor.policy.state_dict())
target_critic.value_func.load_state_dict(online_critic.value_func.state_dict())

optim = torch.optim.Adam(online_critic.value_func.parameters(),lr=ALPHA)


'''Main Loop'''

    
state_v = torch.tensor([env.reset()], dtype=dtype)

for i in count():
#    if i % 1000 == 0:
#        noise.reset()
    env.render()
    #get the action using online actor network and add noise
    action = online_actor.get_action(state_v) + noise.step()
    #get next state, reward, done, and info by taking a step in environment
    next_state, reward, done, _ = env.step(action.detach())
    #convert next state and reward to tensors
    next_state_v = torch.tensor([next_state],dtype=dtype)
    reward_v = torch.tensor([reward],dtype=dtype)
    
    #save the values in the replay buffer
    buffer.push(state_v,action,reward_v,next_state_v,done)
    state_v = next_state_v
    

    #if the buffer is filled up start mini batch sampling and training    
    if len(buffer) < BUFFER_SIZE:
        print('Filling up buffer: %.2f'%(len(buffer)/BUFFER_SIZE*100))
    else:
        #get a mini batch from the replay buffer
        sample = buffer.sample(BATCH_SIZE)
        #make the data nice
        compressed_states, compressed_actions, compressed_next_states, compressed_rewards = utils.extract_data(sample)
        
        #critic network training
        #yt=r(st,at)+γ⋅Q(st+1,μ(st+1))
        na_from_tactor = target_actor.get_action(compressed_next_states)
        v_from_tcritic = target_critic.get_state_value(compressed_next_states, na_from_tactor)
        
        #calculate yt=r(st,at)+γ⋅Q(st+1,μ(st+1))
        target_v = compressed_rewards.unsqueeze(1) + GAMMA * v_from_tcritic
        actual_v = online_critic.get_state_value(compressed_states,compressed_actions)
        loss = nn.MSELoss()
        output = loss(actual_v, target_v)
        optim.zero_grad()
        output.backward(retain_graph=True)
        optim.step()
        
        online_critic.value_func.zero_grad()
        
        for s,a in zip(compressed_states.split(1),compressed_actions.split(1)):
            online_v = online_critic.get_state_value(s,a)
            grad_wrt_a = torch.autograd.grad(online_v,(s,a))
            
            action = online_actor.get_action(s)
            action.backward(retain_graph=True)
            
            for param in online_actor.policy.parameters():
                param.data += ALPHA * (param.grad * grad_wrt_a[1].item())/(BATCH_SIZE)
            
            writer.add_scalar('Action Gradients', grad_wrt_a[1], i)
            online_actor.policy.zero_grad()
            online_critic.value_func.zero_grad()
        
        
#            #soft update
        
        for param_o, param_t in zip(online_actor.policy.parameters(), target_actor.policy.parameters()):
            param_t.data = param_o.data * TAU + param_t.data * (1 - TAU)
            
        
        for param_o, param_t in zip(online_critic.value_func.parameters(), target_critic.value_func.parameters()):
            param_t.data = param_o.data * TAU + param_t.data * (1 - TAU)
        
        online_actor.policy.zero_grad()
        target_actor.policy.zero_grad()
        online_critic.value_func.zero_grad()
        target_critic.value_func.zero_grad()
        
        writer.add_scalar('Critic Loss', output, i)
        writer.add_scalar('Current Reward', reward, i)
        
    
    if i % 5000 == 0:
        # stats
#        evalu_steps = 100
#        state = torch.tensor([env.reset()], dtype=dtype)
#        cumReward = 0
#        for st in range(evalu_steps):
#            a = target_actor.get_action(state)
#            ns, r, d, _ = env.step([action.detach()])
#            cumReward += r
#            state = torch.tensor([ns], dtype=dtype)
#        
#        writer.add_scalar('Target Policy Reward', cumReward, i)
        
        torch.save(target_actor.policy.state_dict(), 'target_actor_state_1.pt')
        torch.save(target_critic.value_func.state_dict(), 'target_critic_state_1.pt')
        