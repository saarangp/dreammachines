#Helmholtz Model V2
import numpy as np
import tensorflow as tf
import math
import torch
import torch.nn as nn

# class Layer(object):

#     def __init__(self, size):
#         self.size = size
#         self.R = np.zeros(size) #Recognition weights
#         self.G = np.zeros(size) #Generative Weights

class helmholtz(object):
    
    def __init__(self, l_sizes, sample_type = 'binomial', epsilon = .1):
        """
        Helmholtz Machine Class w/ k layers
        @param layers (list): list of sizes of layers
        """
        self.layers = []
        for i, size in enumerate(l_sizes):
            if i < len(l_sizes) - 1:
                l_size = (size, l_sizes[i+1])
            else:
                l_size = (size, 1)
            self.layers.append(nn.Linear(l_size[0], l_size[1]))
            print(l_size)

        self.dreams = []
        self.B_G = np.zeros((1,1))
        self.sample_type = sample_type
        self.epsilon = epsilon
    
    def wake_generation(self):
        return

    def wake_recognition(self):
        return
    
    def wake_phase(self):
        return