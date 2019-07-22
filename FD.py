import numpy as np
from numpy.core.umath_tests import inner1d

# TODO: implement second-order finite difference table when mu_2 is not found
# TODO: fd_gradient outputs h as np.array???

class MaxHessian(object):
    """Estimate the maximum absolute Hessian entry for function f"""
    def __init__(self, f, eps, tau_1 = 100, tau_2 = 0.1):
        """
        @param f:         noisy function to be evaluated
        @param eps:       estimated noise level
        @param tau_i:     constant used for Hessian estimation, see More and Wild "Estimating Derivatives of Noisy Simulations", 2012
        """
        self.f = f
        self.eps = eps
        self.tau_1 = tau_1
        self.tau_2 = tau_2

    def calculate_delta(self, x, h, direction = None):
        """Calculate the second-order difference"""
        fx = self.f(x)
        if isinstance(x, (list, np.ndarray)):
            x = np.array(x)
            d = direction if direction is not None else np.random.choice(len(x))
            tmp = np.zeros(len(x))
            tmp[d] = h
            h = tmp
        fx_plus_h = self.f(x + h)
        fx_minus_h = self.f(x - h)
        delta = abs(fx_plus_h + fx_minus_h - 2 * fx)
        return delta, fx, fx_plus_h, fx_minus_h

    def exit(self, h, dh, fx, fx_plus_h, fx_minus_h):
        """Verify if the exit condition has been satisfied
        @param h:          step size
        @param dh:         delta_h: the second-difference calculated by calculate_delta(x, h)
        @param fx:         function value evaluated at x, i.e., f(x)
        @param fx_plus_h:  function value evaluated at f(x+h)
        @param fx_minus_h: function value evaluated at f(x-h)
        """
        c1 = dh / self.eps >= self.tau_1
        c2 = abs(fx_plus_h - fx)  <= self.tau_2 * max(fx, fx_plus_h)
        c3 = abs(fx_minus_h - fx) <= self.tau_2 * max(fx, fx_minus_h)
        return c1 and c2 and c3

    def calculate_mu(self, x, mu = 1, direction = None):
        """Calculate the Hessian estimate
        @param mu: previous estimate of Hessian, to be used as a scalar (default: 1)
        """
        h = (self.eps / mu) ** 0.25
        dh, fx, fx_plus_h, fx_minus_h = self.calculate_delta(x, h, direction)
        mu = dh / h ** 2
        return h, dh, mu, fx, fx_plus_h, fx_minus_h

    def estimate(self, x, direction = None):
        """Estimate the max absolute Hessian
        @param x:         number or list type. Input value
        @param direction: nonnegative integer or None. Direction with respect to
                          which the finite-difference Hessian estimation will be
                          conducted (default: None, random direction)
        """
        h_a, dh_a, mu_a, fx, fx_plus_h_a, fx_minus_h_a = self.calculate_mu(x, direction = direction)
        if self.exit(h_a, dh_a, fx, fx_plus_h_a, fx_minus_h_a):
            return mu_a
        h_b, dh_b, mu_b, fx, fx_plus_h_b, fx_minus_h_b = self.calculate_mu(x, mu = mu_a, direction = direction)
        if self.exit(h_b, dh_b, fx, fx_plus_h_b, fx_minus_h_b) or abs(mu_a - mu_b) <= mu_b / 2:
            return mu_b
        else:
            return 2.0

def f(x):
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    n, d = x.shape
    return inner1d(x, x) + np.random.uniform(1e-3,1e-3)

def fd_gradient(f, x, eps,h = None, mode = "fd", tau_1 = 100, tau_2 = 0.1):
    mu = MaxHessian(f, eps, tau_1, tau_2).estimate(x, direction = None)
    dim = len(x)
    if mu is None: return None
    if mode == "fd":
        fx = f(x)
        if h is None:
            h = 8 ** 0.25 * (eps / mu) ** 0.5
        #h_mat = np.diag(np.ones(dim) * h) TODO vectorize
        f_incr = []
        for d in range(dim):
            tmp = np.zeros(dim)
            tmp[d] = h
            f_incr.append(f(x + tmp))
        f_incr = np.array(f_incr)
        f_decr = fx
    else:
        if h is None:
            h = 3 ** (1/3) * (eps / mu) ** (1/3)
        h_mat = np.zeros((dim, dim))
        h_mat[np.arange(dim), np.arange(dim)] = h
        f_incr, f_decr = f(x + h_mat), f(x - h_mat)
        h *= 2
    #print("mu, h", mu, h)
    return (f_incr - f_decr) / h, h

if __name__ == "__main__":
    print(fd_gradient(f, np.array([1,1]),0.01))


# if __name__ == "__main__":
#     lst = []
#     for _ in range(1000):
#         tmp = fd_gradient(f, np.array([5, 1]), 0.01)
#         if tmp is not None: lst.append(tmp[0])
#     lst = np.array(lst)
#     print(np.mean(lst, axis=0))
