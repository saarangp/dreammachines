#Helmholtz Model V2
import numpy as np
import tensorflow as tf
import math

class Layer(object):

    def __init__(self,size):
        self.size = size
        self.R = np.zeros(size) #Recognition weights
        self.G = np.zeros(size) #Generative Weights

class helmholtz(object):
    
    def __init__(self, l_sizes, sample_type = 'binomial', epsilon = .1):
        """
        Helmholtz Machine Class w/ k layers
        @param layers (list): [782,32,2]
        """
        self.layers = []
        for size in l_sizes:
            self.layers.append(Layer(size))
        print(self.layers)
        self.dreams = []
        self.B_G = np.zeros(l_sizes[0])
        self.sample_type = sample_type
        self.epsilon = epsilon
        


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample(self, p):
        #Takes in probability p and outputs a sample from a distribution over
        type = self.sample_type
        try:
            if p == 0:
                p = 1e-6
                print("Had to make smol number")
        except:
            p[p==0] = 1e-6
        if type == 'binomial':
            return np.random.binomial(1,p)
        if type == 'beta':
            return np.random.beta(p, 1)

    def wake_phase(self, X):

        # output = X
        outputs = [X]
        for layer in self.layers:
            #Recognition
            # print(outputs[-1])
            # print(layer.R)
            sig = self.sigmoid(np.dot(layer.R,outputs[-1]))
            
            # output = self.sample(sig)
            outputs.append(self.sample(sig))

        #Generative
        zeta = (self.sigmoid(self.B_G))
        self.B_G += self.epsilon * (outputs[-1] - zeta)
        
        for i, layer in enumerate(self.layers):
            delta = self.sigmoid(np.dot(layer.G, outputs[i+1]))
            layer.G += self.epsilon * np.dot(outputs[i] - delta, outputs[i+1])
            
        
    def sleep_phase(self):
        p = (self.sigmoid(self.B_G))

        #DREAM!
        outputs = [self.sample(p)]
        for layer in self.layers[::-1]:
            p = (self.sigmoid(np.dot(layer.G, outputs[-1])))
            outputs.append(self.sample(p))
        
        self.dreams.append(outputs[0])
        #W_R recent output
        for i,layer in enumerate(self.layers[::-1]):
            psi = self.sigmoid(np.dot(layer.R, outputs[i+1]))
            layer.R += self.epsilon * np.dot(outputs[i] - psi, outputs[i+1])
            
    def train(self, X, n_iter = 1000):
        # todo Implement KL Divergence Stopping
        i = 0
        while i < n_iter:
            self.wake_phase(X)
            self.sleep_phase()
            i+=1




        