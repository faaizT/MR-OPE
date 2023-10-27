from re import A
import xdrlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import pathlib
import os
import pickle
from tqdm import tqdm
import pdb
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
from utils.helper_functions import Gaussian

class Policy:
    """Defining the action to take based on the value of x, i.e. the action goes from 1->4 based on the std we are in

    Args:
        x (torch.Tensor): this is the context # size n x 1

    Returns:
        [torch.Tensor]: One-hot encoded tensor of the action
    """

    def __init__(self, noise, num_actions):
        self.noise = noise
        self.num_actions = num_actions

    def __call__(self, x):
        # TODO this is really hacky atm
        # add some stochasticity
        actions_probs = torch.ones(x.shape[0], self.num_actions) * self.noise

        log_bool1 = torch.abs(x) > 3
        log_bool2 = torch.abs(x) <= 3
        actions_probs[log_bool1.reshape(-1,), 0] = 1-self.noise
        actions_probs[log_bool2.reshape(-1,), 1] = 1-self.noise

        return actions_probs
        

    def sample(self, x):
        actions_probs = self.__call__(x)
        m = Categorical(probs=actions_probs)
        actions = F.one_hot(m.sample())
        return actions


class DeterministicPolicy(Policy):

    def __init__(self, action, num_actions):
        self.action_to_choose = action
        self.num_actions = num_actions

    def __call__(self, x):
        actions_probs = torch.zeros(x.shape[0], self.num_actions)
        actions_probs[:, self.action_to_choose] = 1
        return actions_probs

class ContinuousPolicy(Policy):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def __call__(self, x, debug=False):
        x_clipped = torch.clip(x, min=-5, max=5)
        actions_probs = torch.zeros(x.shape[0], self.num_actions)
       
        # actions_probs[:, 0] = (x_clipped**2 + 1).reshape(-1)/((x_clipped**2 + 1).max())
        actions_probs[:, 0] = torch.clip(torch.abs(x_clipped/3).reshape(-1)/((torch.abs(x_clipped/3 )).max() + 1) + 0.1, 0.2, 0.8)
        actions_probs[:, 1] = 1 - actions_probs[:, 0]
        if debug:
            import matplotlib.pyplot as plt
            plt.scatter(x_clipped, actions_probs[:, 0])
            plt.savefig('actions_vs_x.png')
            plt.close()
        return actions_probs
    
    
class ContinuousActionPolicy(Policy):
    def __init__(self, type_pol='gaussian'):
        self.type_pol = type_pol

    def __call__(self, x, a, debug=False):
        actions_probs = torch.zeros(x.shape[0], 1)
        if self.type_pol == 'gaussian':
            actions_probs = Gaussian(x, 1)(a)
        elif self.type_pol == 'gaussian2':
            actions_probs = Gaussian(x + 2, 1)(a)
        return actions_probs
            
    def sample(self, x):
        if self.type_pol == 'gaussian':
            actions = torch.normal(mean=x, std=1)
        elif self.type_pol == 'gaussian2':
            actions = torch.normal(mean=x + 2, std=1)
        else:
            raise NotImplementedError
        return actions

    def density_value(self, x, a):
        return self.__call__(x, a)
