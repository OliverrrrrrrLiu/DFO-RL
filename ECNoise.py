import numpy as np
from copy import deepcopy
from numpy.linalg import norm
import warnings

class ECNoise(object):
    def __init__(self, f, h = 1e-6, breadth = 3, max_iter = 10):
        self.f = f
        self.breadth = breadth
        self.total = breadth * 2 + 1
        self.h = h
        self.max_iter = max_iter

    def init_fvals_(self, x, direction = None):
        if direction is None: direction = np.random.randn(len(x)) 
        direction /= norm(direction)
        h = self.h * direction
        return np.array([self.f(x + i * h) for i in range(-self.breadth, self.breadth + 1)])

    def noise_estimate(self, x, direction = None):
        fvals = self.init_fvals_(x, direction)
        fmin, fmax = np.min(fvals), np.max(fvals)
        if (fmax-fmin) / max(abs(fmin), abs(fmax)) > 0.1:
            return None, None, 3
        fvals_ = deepcopy(fvals)
        gamma = 1.0
        levels, dsgns = np.zeros(self.total-1), np.zeros(self.total-1)
        for j in range(self.total-1):
            for i in range(self.total-j-1):
                fvals_[i] = fvals_[i+1] - fvals_[i]
            if j == 0 and np.sum(fvals_[0:self.total-1] == 0) >= self.total / 2:
                return None, None, 2
            gamma *= 0.5 * ((j+1) / (2 * (j+1)-1))
            levels[j] = np.sqrt(gamma * np.mean(fvals_[:self.total-j-1]**2))
            emin, emax = np.min(fvals_[:self.total-j-1]), np.max(fvals_[:self.total-j-1])
            if emin * emax < 0.0: dsgns[j] = 1 
        for k in range(self.total-3):
            emin, emax = np.min(levels[k:k+3]), np.max(levels[k:k+3])
            if emax <= 4 * emin and dsgns[k]:
                noise = levels[k]
                return levels[k], levels, 1
        return None, None, 3

    def estimate(self, x, direction = None):
        for i in range(self.max_iter):
            noise, levels, inform = self.noise_estimate(x, direction)
            if inform == 1:
                return noise
            scale = 100 if inform == 2 else 1 / 100
            self.h *= scale
        warnings.warn("Cannot estimate a noise level from {} iterations".format(self.max_iter))
        return noise

def f(x):
    return np.inner(x,x) * (1 + np.random.normal(0, 1e-4))

if __name__ == "__main__":
    x = np.array([1, 1])
    h = 1e-8
    ecn = ECNoise(f, h=h, breadth=7, max_iter=100)
    print(ecn.estimate(x))