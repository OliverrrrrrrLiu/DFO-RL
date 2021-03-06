import numpy as np
from copy import deepcopy
from numpy.linalg import norm
import warnings

class ECNoise(object):
    """
    Determines the noise of a function from the function values at equally-spaced points
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
        if (fmax-fmin) / (max(abs(fmin),abs(fmax))) > 0.1: #Compute the range of function values; h is too large if min and max differ too much
            print("first condition failed")
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
            levels[j] = np.sqrt(gamma * np.mean(fvals_[:self.total-j-1]**2)) # compute the estimates for the noise level
            emin, emax = np.min(fvals_[:self.total-j-1]), np.max(fvals_[:self.total-j-1]) #Determine differences in sign
            if emin * emax < 0.0: dsgns[j] = 1 
        for k in range(self.total-3): #Determine the noise level
            emin, emax = np.min(levels[k:k+3]), np.max(levels[k:k+3])
            if emax <= 4 * emin and dsgns[k]:
                noise = levels[k]
                return levels[k], levels, 1
        return 0, None, 3 # If noise not detected, then h is too large

    def estimate(self, x, direction = None):
        """
        Estimate the noise level of a noisy function in max_iter iterations

        @param x: the point at which noise estimation is performed
        @param direction: the direction at which noise estimation is performed.
                         If the argument direction isn't passed in, the default direction None is used and a random direction is sampled from uniform distribution
        
        @return noise: the noise level of function
        @warning: if noise is not detected after max_iter runs a warning will pop out and noise is returned as 0
        """
        for i in range(self.max_iter):
            noise, levels, inform = self.noise_estimate(x, direction) 
            if inform == 1: #if noise is detected
                return noise, levels, inform, self.total
            scale = 100 if inform == 2 else 1 / 100 #if noise is not detected, modify h according to inform 
            self.h *= scale
        warnings.warn("Cannot estimate a noise level from {} iterations".format(self.max_iter))
        print(noise)
        return noise, levels, inform, self.total


"""
Test case
"""
def f(x):
    ###########Rosenbrock##################
    #return (100*(x[1]-x[0]**2)**2 + (1-x[0])**2 ) * (1 + 1e-2 * np.random.normal(0,1)) 
    #return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2  + 1e-23 * np.random.normal(0,1)
    ###########Linear######################
    return 4*x[0] + 10000000000000000*x[1] + 1e-5*np.random.normal(0,1)
    #return (4*x[0] + 5*x[1]) * (1 + 1e-15*np.random.normal(0,1))
    ###########Quadratic###################
    #return np.inner(x,x) + np.random.uniform(-1e-3,1e-3)
    #return np.inner(x,x) + 1e-4*np.random.randn()
    #return (np.inner(x,x) )*(1+ 1e-4*np.random.randn())
    ###########Constant####################
    #return 1e-23 * np.random.normal(0,1)
    ###########Cos/Sin#####################
    #return np.cos(100*x) + np.sin(x) + 1e-2 * np.random.uniform(0,2*np.sqrt(3))
    #return (np.cos(100*x) + np.sin(x)) * (1 + 1e-2 * np.random.uniform(0,2*np.sqrt(3)))
    ###########tangent#####################
    #return np.tan(x) + 1e-5*np.random.normal(0,1)
    #return np.tan(x) * (1 + 1e-5*np.random.normal(0,1))
    #_x = np.zeros(len(x))
    #_x[:-1] = x[1:]
    #res = np.sum(100 * (_x[:-1] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2) *(1 + 1e-2 * np.random.rand())
    #return res

if __name__ == "__main__":
    #x = np.array([np.pi/2 - 1e-15])
    x = np.array([1,1])
    #x = np.array([1,1,1,1,1])
    h = 1e-6
    ecn = ECNoise(f, h = h, breadth = 3, max_iter = 10)
    noise, levels, informs,evals = ecn.estimate(x)
    print(noise, informs)

# def f(x):
#     return np.inner(x,x) + np.random.uniform(-1e-3,1e-3)

# if __name__ == "__main__":
#     x = np.array([-0.01529986, -0.00816587]) 
#     h = 1e-8
#     ecn = ECNoise(f, h=h, breadth=7, max_iter=100)
#     print(ecn.estimate(x))

"""
def f(x):
    return np.inner(x,x) + x[0] + np.random.uniform(-1e-3,1e-3)

if __name__ == "__main__":
    x = np.array([-0.5, 0.0]) 
    h = 1e-8
    ecn = ECNoise(f, h=h, breadth=7, max_iter=100)
    print(ecn.estimate(x))
"""