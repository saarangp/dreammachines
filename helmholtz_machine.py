import numpy as np
import tensorflow as tf
import math

class helmholtz(object):
    def __init__(self, epsilon, size):
        self.V_R = np.zeros(size) #TODO: Fix weights
        self.W_R = np.zeros(size) #TODO: Fix This
        self.V_G = np.zeros(size) #TODO: Fix weights
        self.W_G = np.zeros(size) #TODO: Fix This
        self.B_G = np.zeros(size) #TODO: Fix
        self.dreams = []
        self.epsilon = epsilon

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample(self, p, type = 'binomial'):
        #Takes in probability p and outputs a sample from a distribution over
        type = 'beta'
        try:
            if p == 0:
                p = 1e-6
                print("Had to make smol number")
        except:
            p[p==0] = 1e-6

        # p = p.numpy()
        if type == 'binomial':
            return np.random.binomial(1,p)
        if type == 'beta':
            return np.random.beta(p, 1)

    def wake_phase(self, X):
        try:
            # Recognition Pass
            p = self.sigmoid(np.dot(self.V_R, X))
            X_layer1 = self.sample(p)
            p = self.sigmoid(np.dot(self.W_R, X_layer1))
            X_layer2 = self.sample(p)

            # Generative Pass
            zeta = (self.sigmoid(self.B_G))
            psi = (self.sigmoid(np.dot(self.W_G, X_layer2)))
            delta = (self.sigmoid(np.dot(self.V_G, X_layer1)))

            self.B_G += self.epsilon * (X_layer2 - zeta)
            self.W_G += self.epsilon * np.dot(X_layer1 - psi, X_layer2)
            self.V_G += self.epsilon * np.dot(X - delta, X_layer1)
        except:
            print(p)
            raise


    def sleep_phase(self):
        p = (self.sigmoid(self.B_G))

        #DREAM!
        X = self.sample(p)
        # self.dreams.append(X)

        p = (self.sigmoid(np.dot(self.W_G, X)))
        X_layer1 = self.sample(p)

        p = (self.sigmoid(np.dot(self.V_G, X_layer1)))
        X_layer2 = self.sample(p)

        self.dreams.append(X_layer2)

        psi = (self.sigmoid(np.dot(self.V_R, X_layer2)))
        zeta = (self.sigmoid(np.dot(self.W_R, X_layer1)))

        self.V_R += self.epsilon * np.dot(X_layer1 - psi, X_layer2)
        self.W_R += self.epsilon * np.dot(X - zeta, X_layer1)

    def train(self, X, n_iter = 1000):
        # todo Implement KL Divergence Stopping
        i = 0
        while i < n_iter:
            self.wake_phase(X)
            self.sleep_phase()
            i+=1


