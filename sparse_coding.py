import numpy as np
from progressbar import progressbar

def get_selections(X, n):
    selections = np.random.choice(np.arange(X.shape[0]), n, replace=False)
    return X[selections]

def calc_g(u, lmbda):
    return np.sign(u) * np.maximum(np.abs(u) - lmbda, 0)

def update_func(Phi, X, G, a):
    return Phi.T @ X.T - G @ a

def calc_LCA(Phi, X, lmbda=0.1, alpha=0.001, num_steps=1000):
    """
    Calculate the activations of an image X given a feature dict Phi
    using gradient descent
    """
    G = Phi.T @ Phi
    G = G - np.eye(G.shape[0])
    
    a = np.zeros((Phi.shape[1], X.shape[0]))
    u = np.zeros((a.shape[0], a.shape[1]))
    prev_u = np.zeros(u.shape)
    for i in range(num_steps):
        u = (1 - alpha) * u + alpha * update_func(Phi, X, G, a)
        a = calc_g(u, lmbda)
        if np.linalg.norm(u - prev_u) < 0.05:
            break
        prev_u = u
    return a

def calc_Phi(X, a_dim, alpha=0.001, num_steps=2000, batch_size=100, saved_phi=None):
    """
    Calculate the feature dictionary for an image set X
    using nested gradient descent.
    """
    Phi = saved_phi
    if Phi is None:   
        Phi = np.random.rand(X.shape[1], a_dim)
        Phi = Phi - np.mean(Phi)
        Phi = Phi @ np.diag(1 / np.linalg.norm(Phi, ord=2, axis=0))
    for i in progressbar(range(num_steps)):
        next_X = get_selections(X, batch_size)
        a = calc_LCA(Phi, next_X)
        Phi = Phi + alpha * (next_X.T - Phi @ a) @ a.T
        Phi = Phi @ np.diag(1 / np.linalg.norm(Phi, ord=2, axis=0))

    return Phi


class SparseCodingModel(object):
    def __init__(self, batch_size, n_activations, alpha):
        self.batch_size = batch_size
        self.n_activations = n_activations
        self.alpha = alpha
        self.Phi = None

    def train(self, X, alpha=0.001, num_steps=2000):
        self.Phi = calc_Phi(X, self.n_activations, alpha=alpha, num_steps=num_steps, 
                            batch_size=self.batch_size, saved_phi=self.Phi)
    
    def predict(self, X, lmbda=0.1, alpha=0.001, num_steps=1000):
        return calc_LCA(self.Phi, X, lmbda, alpha, num_steps)

    def generate(self, A):
        return np.dot(self.Phi, A)

    def save_Phi(self, file_name):
        np.save(file_name, self.Phi)

    def load_Phi(self, file_name):
        self.Phi = np.load(file_name)
