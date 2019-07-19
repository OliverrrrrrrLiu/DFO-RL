import numpy as np
from ECNoise import ECNoise
from FD import fd_gradient
from LBFGS import LBFGS
from linesearch import LineSearch
from numpy.linalg import norm

class FDLM(object):
    def __init__(self, f, ecn_params, ls_params, zeta, m):
        """
        @param ecn_params: tuple of ECNoise parameters (h, breadth, max_iter)
        @param ls_params: tuple of line search parameters (f, c1, c2, gamma_1, gamma_2, max_iter)
        @param m: length of L-BFGS history
        """
        self.zeta = zeta
        self.eval_counter = 0
        self.ls_counter = 0
        self.rec_counter = 0
        self.noise_f = ECNoise(f, *ecn_params)
        self.ls = LineSearch(f, *ls_params)
        self.lbfgs = LBFGS(m)

    def run(self, f, x):
        f_val = f(x)
        noise = self.noise_f.estimate()
        grad, h = fd_gradient(f, x, noise)
        # TODO: Stencil
        k = 0
        while not self.is_convergence():
            d = self.lbfgs.calculate_direction(grad)
            new_pt, step, TODO, flag = self.ls.search((x, f_val, grad), d)
            if not flag:
                new_pt, h, noise, TODO = self.ls.recovery((x, f_val, grad), h, d, stencil_pt)
            else:
                h = None
            x_new, f_val_new = new_pt
            grad_new, h_new = fd_gradient(f, x_new, noise, h)
            s, y = x_new - x, grad_new - grad
            if self.curvature_satisfied(s, y):
                self.lbfgs.update_history(s, y)
            k += 1

    def curvature_satisfied(self, s, y, f_val, f_MA):
        return np.inner(s, y) >= self.zeta * norm(s) * norm(y)

    def is_convergence(self, grad, tol = 1e-8):
        return norm(grad) <= tol or abs(f_MA - f_val) <= tol