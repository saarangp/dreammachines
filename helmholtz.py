#Helmholtz Model V2

import numpy as np
import math

class Layer(object):

    def __init__(self, size):
        self.size = size
        self.R = np.zeros(size) #Recognition weights
        self.G = np.zeros(size) #Generative Weights

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
            self.layers.append(Layer(l_size))
            print(l_size)

        self.dreams = []
        self.B_G = np.zeros((1,1))
        self.sample_type = sample_type
        self.epsilon = epsilon

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample(self, p):
        #Takes in probability p and outputs a sample from a distribution over
        dist_type = self.sample_type
        try:
            if p == 0:
                p = 1e-6
                print("Had to make smol number")
        except:
            p[p==0] = 1e-6
        # if len(p)==1:
        #     p = 1e-6
        # else:
        #     p[p==0] = 1e-6
        if dist_type == 'binomial':
            return np.random.binomial(1,p)
        if dist_type == 'beta':
            return np.random.beta(p, 1)

    def wake_phase(self, X):

        # output = X
        #Recognition
        outputs = [X]
        # print(f"X.shape: {X.shape}")
        for layer in self.layers:
            sig = self.sigmoid(np.dot(outputs[-1], layer.R))
            # print("recognition: ", sig.shape, outputs[-1].shape, layer.R.shape)
            outputs.append(self.sample(sig))

        #Generative
        zeta = (self.sigmoid(self.B_G))
        self.B_G += self.epsilon * (outputs[-1] - zeta)
        
        for i, layer in enumerate(self.layers):
            # print("Generative a:", outputs[i+1].shape, layer.G.shape, outputs[i].shape)
            delta = self.sigmoid(np.dot(outputs[i+1], layer.G.T))
            # print("Generative b:", outputs[i + 1].shape, layer.G.shape, outputs[i].shape, delta.shape)
            layerG_upd = self.epsilon * np.dot(outputs[i + 1].T, outputs[i] - delta)
            # print(f"layerG Updated: {layerG_upd.shape}")
            layer.G += layerG_upd.T
            
    def sleep_phase(self):
        p = (self.sigmoid(self.B_G))

        #DREAM!
        outputs = [self.sample(p)]
        for layer in self.layers[::-1]:
            p = (self.sigmoid(np.dot(layer.G, outputs[-1])))
            
            outputs.append(self.sample(p))
        
        self.dreams.append(outputs[-1])
        #W_R recent output
        for i,layer in enumerate(self.layers[::-1]):
            #print("sleep: ", layer.R.shape, outputs[i+1].shape)
            psi = self.sigmoid(np.dot(outputs[i + 1].T, layer.R))
            
            #print("psi: ", psi.shape, "outputs[i]: ", outputs[i].shape,  "outputs[i+1]: ", outputs[i+1].shape)
            
            layerR_upd = self.epsilon * np.dot(outputs[i+1], outputs[i].T - psi)
            #print(f"layerR Updated: {layerR_upd.shape}")
            layer.R += layerR_upd
            
    def train(self, X, n_iter = 1000):
        # todo Implement KL Divergence Stopping
        i = 0
        while i < n_iter:
            self.wake_phase(X)
            self.sleep_phase()
            i+=1




class SparseHelmholtz(helmholtz):
    def __init__(self, l_sizes, sample_type = 'binomial', epsilon = .1, lmbda=0.1):
        super().__init__(l_sizes, sample_type, epsilon)
        self.lmbda = lmbda

    def wake_phase(self, X):
        l1_terms = [] # calculate l1 gradient
        for i, layer in enumerate(self.layers[::-1]):
            layer_l1 = self.lmbda * np.sign(layer.G) 
            l1_terms.append(layer_l1)
            
        super().wake_phase(X)
        # iterate over new layers and add l1 gradient
        for i, layer in enumerate(self.layers[::-1]):
            layer.G -= l1_terms[i]

    def sleep_phase(self):
        l1_terms = [] # calculate l1 gradient
        for i, layer in enumerate(self.layers[::-1]):
            layer_l1 = self.lmbda * np.sign(layer.R) 
            l1_terms.append(layer_l1)

        super().sleep_phase()
        # iterate over new layers, add l1 gradient
        for i, layer in enumerate(self.layers[::-1]):
            layer.R -= l1_terms[i]
        