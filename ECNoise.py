import numpy as np
import warnings

#TODO: incorporate random directions in function evaluations

class ECNoise(object):
    """Estimate the noise level of function f from function values evaluated at equally spacing points
    @param h: spacing
    @param breadth: (number of points-1)/2
    """
    def __init__(self, f, x, h, breadth = 3, max_iter = 10, fval = None):
        self.f = f
        self.x = x
        self.breadth = breadth
        self.total = breadth * 2 + 1
        self.h = h
        self.fval = fval if fval is not None else f(x)
        self.fvals = self.init_fvals()
        self.max_iter = max_iter

    
    def init_fvals(self):
        #Random direction p
        dim = len(self.x)
        vec = np.random.randn(dim)
        p = vec/np.linalg.norm(vec)
        h = self.h * p
        #return np.array([self.f(self.x - i * self.h) for i in range(self.breadth, 0, -1)] + [self.fval] +
        #                [self.f(self.x + i * self.h) for i in range(1, self.breadth + 1)])
        return np.array([self.f(self.x - i * h) for i in range(self.breadth, 0, -1)] + [self.fval] +
                        [self.f(self.x + i * h) for i in range(1, self.breadth + 1)])

    def estimate_given_h(self):
        """
        inform = 1  Noise has been detected.
        inform = 2  Noise has not been detected; h is too small.
        inform = 3  Noise has not been detected; h is too large.
        """
        fmax, fmin = np.max(self.fvals), np.min(self.fvals)
        if (fmax - fmin) / max(abs(fmax), abs(fmin)) > 0.1:
            return None, None, 3

        fvals_ = np.copy(self.fvals)
        gamma = 1.0
        levels, dsgns = np.zeros(self.total - 1), np.zeros(self.total - 1)

        for j in range(self.total-1):
            for i in range(self.total-j-1):
                fvals_[i] = fvals_[i+1] - fvals_[i]
            if j == 0 and np.sum(fvals_[0:self.total-1] == 0) >= self.total / 2:
                return None, None, 2
            gamma *= 0.5 * ((j + 1)/(2 * (j + 1) - 1))
            levels[j] = np.sqrt(gamma * np.mean(fvals_[0:self.total-j-1] ** 2))
            emax, emin = np.max(fvals_[0:self.total-j-1]), np.min(fvals_[0:self.total-j-1])
            if emax * emin < 0.0:
                dsgns[j] = 1

        for k in range(self.total-3):
            emax, emin = np.max(levels[k:k+3]), np.min(levels[k:k+3])
            if emax <= 4 * emin and dsgns[k]:
                noise = levels[k]
                return levels[k], levels, 1
        return None, None, 3

    def estimate(self):
        for i in range(self.max_iter):
            noise, levels, inform = self.estimate_given_h()
            if inform == 1:
                return noise
            scale = 100 if inform == 2 else 1 / 100
            self.h *= scale
            self.fvals = self.init_fvals()
        warnings.warn("Cannot estimate a noise level from {} iterations".format(self.max_iter))
        return noise

def f(x):
    return np.inner(x,x) * (1 + 1e-3*np.random.normal(0, 1))

if __name__ == "__main__":
    x = np.array([1, 1])
    h = 1e-6
    ecn = ECNoise(f, x, h)
    print(ecn.estimate())

"""
def estimate_noise(nf,fval):
    # Determines the noise of a function from the function values

    # The user must provide the function value at nf equally-spaced points.
    # For example, if nf = 7, the user could provide

    #    f(x-3h), f(x-2h), f(x-h), f(x), f(x+h), f(x+2h), f(x+3h)

    # in the array fval. Although nf >= 4 is allowed, the use of at least
    # nf = 7 function evaluations is recommended.

    # Noise will not be detected by this code if the function values differ
    # in the first digit.

    # If noise is not detected, the user should increase or decrease the
    # spacing h according to the ouput value of inform.  In most cases,
    # the subroutine detects noise with the initial value of h.

    # On exit:
    #   fnoise is set to an estimate of the function noise;
    #      fnoise is set to zero if noise is not detected.

    #   level is set to estimates for the noise. The k-th entry is an
    #     estimate from the k-th difference.

    #   inform is set as follows:
    #     inform = 1  Noise has been detected.
    #     inform = 2  Noise has not been detected; h is too small.
    #                 Try 100*h for the next value of h.
    #     inform = 3  Noise has not been detected; h is too large.
    #                 Try h/100 for the next value of h.

    #    Argonne National Laboratory
    #    Jorge More' and Stefan Wild. November 2009

    level = np.zeros((nf-1, 1))
    dsgn = np.zeros((nf-1,1))
    fnoise = 0.0
    gamma = 1.0

    #Compute the range of function values
    fmin = np.min(fval)
    fmax = np.max(fval)
    if (fmax - fmin)/np.max(np.abs(fmax), np.abs(fmin)) >.1:
        inform = 3
        return fnoise, level, inform

    #Construct the difference table
    for j in range(nf-1):
        for i in range(nf-j-1):
            fval[i] = fval[i+1] - fval[i]
        #h is too small only when half the function values are equal
        if j==0 and np.sum(fval[0:nf-1] == 0) >= nf/2:
            inform = 2
            return fnoise,level,inform

        gamma = 0.5 * ((j+1)/(2*(j+1) - 1)) * gamma

        #Compute the estimates for the noise level
        level[j] = np.sqrt(gamma*np.mean(fval[0:nf-j-1]**2))

        #Determine differences in sign
        emin = np.min(fval[0:nf-j-1])
        emax = np.max(fval[0:nf-j-1])
        if emin*emax < 0.0:
            dsgn[j] = 1

    #Determine the noise level
    for k in range(nf-3):
        emin = np.min(level[k:k+3])
        emax = np.max(level[k:k+3])
        if emax <= 4*emin and dsgn[k]:
            fnoise = level[k]
            inform = 1
            return fnoise, level, inform
    #If noise not detected then h is too large
    inform = 3
    return fnoise, level, inform
"""
