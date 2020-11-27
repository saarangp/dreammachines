import numpy as np

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
    for i in range(num_steps):
        if i % 100 == 0:
            print(i)
        a = calc_LCA(Phi, X)
        Phi = Phi + alpha * (X.T - Phi @ a) @ a.T
        Phi = Phi @ np.diag(1 / np.linalg.norm(Phi, ord=2, axis=0))

    return Phi