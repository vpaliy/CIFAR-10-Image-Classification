import numpy as np


class LassoRegularization(object):
    __slots__ = ('alpha',)
    
    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        
    def __call__(self, weights):
        return self.alpha * np.linalg.norm(weights)
    
    def grad(self, weights):
        return self.alpha * np.sign(weights)


class RidgeRegularization(object):
    def __init__(self, alpha=1e-3):
        self.alpha = alpha
    
    def __call__(self, weights):
        return self.alpha * np.sum(weights.T.dot(weights)) * 0.5
    
    def grad(self, weights):
        return self.alpha * weights


class ElasticNet(object):
    def __init__(self, alpha=1e-3, r=1):
        self.alpha = alpha
        self.r =r 
    
    def __call__(self, weights):
        l1, l2 = self.r * np.linalg.norm(weights), (1-self.r)*0.5 * np.sum(weights.T.dot(weights))
        return self.alpha * (l1 + l2)
    
    def grad(self, weights):
        l1, l2 = self.r * np.sign(weights), (1-self.r) * weights
        return self.alpha * (l1 + l2)
