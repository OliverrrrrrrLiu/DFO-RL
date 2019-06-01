import numpy as np

class MaxHessian(object):
    """Estimate the maximum absolute Hessian entry for function f"""
    def __init__(self, f, eps, tau_1 = 100, tau_2 = 0.1):
        """
        @param f:     noisy function to be evaluated
        @param eps:   estimated noise level
        @param tau_i: constant used for Hessian estimation, see More and Wild "Estimating Derivatives of Noisy Simulations", 2012
        """
        self.f = f
        self.eps = eps
        self.tau_1 = tau_1
        self.tau_2 = tau_2

    def calculate_delta(self, x, h):
        """Calculate the second-order difference
        @param x: input value of our function f
        @param h: step size
        """
        fx = self.f(x)
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
        c2 = abs(fx_plus_h - fx)  <= self.tau_2 * max(fx, fx_plus_h) #TODO vectorize
        c2 = abs(fx_minus_h - fx) <= self.tau_2 * max(fx, fx_minus_h)
        return c1 and c2 and c3

    def calculate_mu(self, x, mu = 1):
        """Calculate the Hessian estimate
        @param x:  input value
        @param mu: previous estimate of Hessian, to be used as a scalar (default: 1)
        """
        h = (self.eps / mu) ** 0.25
        dh, fx, fx_plus_h, fx_minus_h = self.calculate_delta(x, h)
        mu = dh / h ** 2
        return h, dh, mu, fx, fx_plus_h, fx_minus_h

    def estimate(self, x):
        """Estimate the max absolute Hessian"""
        h_a, dh_a, mu_a, fx, fx_plus_h_a, fx_minus_h_a = self.calculate_mu(x)
        if self.exit(h_a, dh_a, fx, fx_plus_h_a, fx_minus_h_a):
            return mu_a
        h_b, dh_b, mu_b, fx, fx_plus_h_b, fx_minus_h_b = self.calculate_mu(x, mu_a)
        if self.exit(h_b, dh_b, fx, fx_plus_h_b, fx_minus_h_b) or abs(mu_a - mu_b) <= mu_b / 2:
            return mu_b
        else:
            return None