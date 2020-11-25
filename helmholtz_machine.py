import numpy as np
import tensorflow as tf

class helmholtz(object):
    def __init__(self, epsilon, size):
        self.V_R = np.zeros(size) #TODO: Fix weights
        self.W_R = np.zeros(size) #TODO: Fix This
        self.V_G = np.zeros(size) #TODO: Fix weights
        self.W_G = np.zeros(size) #TODO: Fix This
        self.B_G = np.zeros(size) #TODO: Fix

        self.epsilon = epsilon

    def wake_phase(self, X):
        # Recognition Pass
        p = tf.asnumpy(tf.math.sigmoid(np.dot(self.V_R, X)))
        X_layer1 = np.random.binomial(1, p)
        p = tf.asnumpy(tf.math.sigmoid(np.dot(self.W_R, X_layer1)))
        X_layer2 = np.random.binomial(1, p)

        # Generative Pass
        zeta = tf.asnumpy(tf.math.sigmoid(self.B_G))
        psi = tf.asnumpy(tf.math.sigmoid(np.dot(self.W_G, X_layer2)))
        delta = tf.asnumpy(tf.math.sigmoid(np.dot(self.W_G, X_layer1)))

        self.B_G += self.epsilon * (X_layer2 - zeta)
        self.W_G += self.epsilon * np.dot(X_layer1 - psi, X_layer2)
        self.V_G += self.epsilon * np.dot(X - delta, X_layer1)


    def sleep_phase(self):
        p = tf.asnumpy(tf.math.sigmoid(self.B_G))
        X = np.random.binomial(1, p)

        p = tf.asnumpy(tf.math.sigmoid(np.dot(self.W_G, X)))
        X_layer1 = np.random.binomial(1, p)

        p = tf.asnumpy(tf.math.sigmoid(np.dot(self.V_G, X_layer1)))
        X_layer2 = np.random.binomial(1, p)

        psi = tf.asnumpy(tf.math.sigmoid(np.dot(self.V_R, X_layer2)))
        zeta = tf.asnumpy(tf.math.sigmoid(np.dot(self.W_R, X_layer2)))

        self.V_R += self.epsilon * np.dot(X_layer1 - psi, X_layer2)
        self.W_R += self.epsilon * np.dot(X - zeta, X_layer1)

    def train(self, X, n_iter = 1000):
        # todo Implement KL Divergence Stopping
        i = 0
        while i < n_iter:
            self.wake_phase(X)
            self.sleep_phase()
            i+=1


