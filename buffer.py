from collections import namedtuple
import random

Experience = namedtuple('Experience',('state','action','reward','next_state','done'))

class Memory(object):
    def __init__(self, mem_size):
        self.capacity = mem_size
        self.memory = []
        self.position = 0
        
    def push(self, s,a,r,ns,d):
        """Save a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            
        self.memory[self.position] = Experience(s,a,r,ns,d)
        self.position = (self.position+1) % self.capacity
        
    
    def sample(self, batch_size = 32):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
            