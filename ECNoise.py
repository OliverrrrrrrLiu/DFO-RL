import numpy as np
from copy import deepcopy
from numpy.linalg import norm
import warnings

class ECNoise(object):
    """
    Determines the nosie of a function from the function values at equally-spaced points
    """
    def __init__(self, f, h = 1e-6, breadth = 3, max_iter = 10):
        """
        Inits ECNoise with functions and finite difference table paramters

        @param f: noisy function f to be evaluated
        @param h: space between the equally-spaced points to be evaluated
        @param breadth: (number of function values - 1)/2
        @param max_iter: maximum number of iterations to perform noise estimation
        """
        self.f = f
        self.breadth = breadth
        self.total = breadth * 2 + 1 #total number of function evaluations
        self.h = h
        self.max_iter = max_iter

    def init_fvals_(self, x, direction = None):
        """
        Evaluate function values in at equally spaced points in a given direction
        For example, if breadth = 3, the function evaluates f(x-3h), f(x-2h), f(x-h), f(x), f(x+h), f(x+2h), f(x+3h)

        @param x: the point at which noise estimation is performed
        @param direction: the direction at which noise estimation is performed.
                         If the argument direction isn't passed in, the default direction None is used and a random direction is sampled from uniform distribution

        @return: an array of function values evaluated at equally spaced points 
        """
        if direction is None: direction = np.random.randn(len(x)) #a random direction with uniform distribution is sampled if no direction is given
        direction /= norm(direction) #normalize direction 
        h = self.h * direction #vectorize spacing h to
        return np.array([self.f(x + i * h) for i in range(-self.breadth, self.breadth + 1)])

    def noise_estimate(self, x, direction = None):
        """
        Determine the noise of the function from the function values from difference table
        If noise is not detected, user should increase or decrease the spacing h according to the output values of inform. 
        In most cases, the subroutine detects noise with the initial value of h

        @param x: the point at which noise estimation is performed
        @param direction: the direction at which noise estimation is performed.
                         If the argument direction isn't passed in, the default direction None is used and a random direction is sampled from uniform distribution
        
        @return noise: estimate of the function noise; is set to 0 if noise is not detected
        @return level: is set to estimate for the noise; the k-th entry is an estimate from the k-th difference
        @return inform: 
            inform = 1 Noise has been detected
            inform = 2 Noise has not been detected; h is too small; Try 100*h for the next value of h
            inform = 3 Noise has not been detected; h is too large; Try h/100 for the next value of h
        """
       #print("x", x)
        fvals = self.init_fvals_(x, direction) #collect the function values at points equally spaced around x 
        fmin, fmax = np.min(fvals), np.max(fvals) # 
        if (fmax-fmin) / max(abs(fmin), abs(fmax)) > 0.1: #Compute the range of function values; h is too large if min and max differ too much
            return 0, None, 3
        fvals_ = deepcopy(fvals) #create a copy of the function values
        gamma = 1.0 # gamma_j = (j!)^2/(2j)!
        levels, dsgns = np.zeros(self.total-1), np.zeros(self.total-1)
        for j in range(self.total-1): # Construct the difference table
            for i in range(self.total-j-1):
                fvals_[i] = fvals_[i+1] - fvals_[i] #calculate the function differences
            if j == 0 and np.sum(fvals_[0:self.total-1] == 0) >= self.total / 2: #h is too small only when half the function values are equal
                return 0, None, 2
            gamma *= 0.5 * ((j+1) / (2 * (j+1)-1))
            levels[j] = np.sqrt(gamma * np.mean(fvals_[:self.total-j-1]**2))
            emin, emax = np.min(fvals_[:self.total-j-1]), np.max(fvals_[:self.total-j-1])
            if emin * emax < 0.0: dsgns[j] = 1 
        for k in range(self.total-3):
            emin, emax = np.min(levels[k:k+3]), np.max(levels[k:k+3])
            if emax <= 4 * emin and dsgns[k]:
                noise = levels[k]
                return levels[k], levels, 1
        return 0, None, 3

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
    return np.inner(x,x) + np.random.uniform(-1e-3,1e-3)

if __name__ == "__main__":
    x = np.array([-0.01529986, -0.00816587]) * 6
    h = 1e-8
    ecn = ECNoise(f, h=h, breadth=7, max_iter=100)
    print(ecn.estimate(x, direction=[-0.70335354 -0.71084021]))