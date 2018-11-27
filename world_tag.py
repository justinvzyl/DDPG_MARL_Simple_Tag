from make_env import make_env
import gym
import torch
import multiprocessing

class World():
    def __init__(self, render = False):
        self.env  = make_env('simple_tag.py')
        self.must_render = render
        
    def agent_hunt(self):
        return None
        
    def agent_avoid(self):
        return None
        
    def train(self):
        return None
        
        
        