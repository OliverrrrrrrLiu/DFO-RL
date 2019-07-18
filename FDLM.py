import numpy as np
from ECNoise import ECNoise
from FD import fd_gradient
from LBFGS import LBFGS
from linesearch import LineSearch
from numpy.linalg import norm

class FDLM(object):
    def __init__(self, f, x0, max_iter, zeta, m, c1, c2, gamma_1, gamma_2, max_iter):
        """
        @param f: noisy function f to be evaluated
        @param x0: initial iterate
        @param max_iter: maximum number of iterations for backtracking line search
        @param zeta: curvature threshold
        @param m: length of L-BFGS history
        """
        self.f = f
        self.max_iter = max_iter
        self.zeta = zeta
        self.eval_counter = 0
        self.ls_counter = 0
        self.rec_counter = 0
        self.x0 = x0
        self.noise_f = ECNoise(f, x0)
        self.lbfgs = LBFGS(m)
        self.ls = LineSearch(f, c1, c2, gamma_1, gamma_2, max_iter)

    def run(self):
        f_val = self.f(self.x0)
        noise = self.noise_f.estimate()
        x = self.x0
        grad = fd_gradient(self.f, x, noise)
        k = 0
        while not self.is_convergence():
            d = self.lbfgs.calculate_gradient(grad)
            new_pt, step, TODO, flag = self.ls.search((x, f_val, grad), d)
            if not flag:
                new_pt, h, noise, TODO = self.ls.recovery((x, f_val, grad), h, d, stencil_pt)
            else:
                h = None
            x_new, f_val_new = new_pt
            grad_new = fd_gradient(self.f, x_new, noise, h)
            s, y = x_new - x, grad_new - grad
            if self.curvature_satisfied(s, y):
                self.lbfgs.update_history(s, y)
            k += 1

    def curvature_satisfied(self, s, y):
        return np.inner(s, y) >= self.zeta * norm(s) * norm(y)

    def is_convergence(self):
        return False

