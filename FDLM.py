import numpy as np

class FDLM(object):
    def __init__(self, f, x0, max_iter, zeta):
        """
        @param f: noisy function f to be evaluated
        @param x0: initial iterate
        @param max_iter: maximum number of iterations for backtracking line search
        @param zeta: curvature threshold
        """
        self.f = f
        self.max_iter = max_iter
        self.zeta = zeta
        self.eval_counter = 0
        self.ls_counter = 0
        self.rec_counter = 0
        self.x0 = x0

    def train(self):
        fx = self.f(self.x0)
        self.eval_counter += 1
