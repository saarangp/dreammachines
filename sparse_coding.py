import numpy as np
from progressbar import progressbar

def calc_g(u, lmbda):
    return np.maximum(u - lmbda, 0)

def update_func(Phi, X, G, a):
    return Phi.T @ X.T - G @ a

def calc_LCA(Phi, X, lmbda=0.1, alpha=0.001, num_steps=1000):
    """
    Calculate the activations of an image X given a feature dict Phi
    using gradient descent
    """
    G = Phi.T @ Phi
    G = G - np.eye(G.shape[0])
    
    a = np.ones((Phi.shape[1], X.shape[0])) / X.shape[0]
    u = np.ones(a.shape) / X.shape[0]

    for i in range(num_steps):
        u = (1 - alpha) * u + alpha * update_func(Phi, X, G, a)
        a = calc_g(u, lmbda)
    return a

def calc_Phi(X, a_dim, alpha=0.001, num_steps=2000):
    """
    Calculate the feature dictionary for an image set X
    using nested gradient descent.
    """
    Phi = np.ones((X.shape[1], a_dim))
    Phi = Phi @ np.diag(1 / np.linalg.norm(Phi, ord=2, axis=0))
    for i in progressbar(range(num_steps)):
        a = calc_LCA(Phi, X)
        Phi = Phi + alpha * (X.T - Phi @ a) @ a.T
        Phi = Phi @ np.diag(1 / np.linalg.norm(Phi, ord=2, axis=0))

    return Phi


class SparseCodingModel(object):
    def __init__(self, input_size, n_activations, alpha):
        self.input_size = input_size
        self.n_activations = n_activations
        self.alpha = alpha

    def train(self, X):
        self.Phi = calc_Phi(X, self.n_activations)
    
    def predict(self, X, num_steps=1000):
        return calc_LCA(self.Phi, X, num_steps=num_steps)

    def generate(self, A):
        return np.dot(self.Phi, A)
