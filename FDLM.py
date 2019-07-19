import numpy as np
from ECNoise import ECNoise
from FD import fd_gradient
from LBFGS import LBFGS
from linesearch import LineSearch
from numpy.linalg import norm
from recovery import Recovery
from numpy.core.umath_tests import inner1d

class FDLM(object):
    def __init__(self, f, ecn_params, ls_params, rec_params, zeta, m):
        """
        @param ecn_params: tuple of ECNoise parameters (h, breadth, max_iter)
        @param ls_params: tuple of line search parameters (c1, c2, max_iter)
        @param rec_params: tuple of recovery parameters (gamma_1, gamma_2)
        @param m: length of L-BFGS history
        """
        self.f = f
        self.zeta = zeta
        self.eval_counter = 0
        self.ls_counter = 0
        self.rec_counter = 0
        self.noise_f = ECNoise(f, *ecn_params)
        self.ls = LineSearch(f, *ls_params)
        self.rec = Recovery(f, *rec_params, self.noise_f, self.ls)
        self.lbfgs = LBFGS(m)

    def get_stencil_pt(self, x, h):
        stencil_pts = []
        dim = len(x)
        for i in range(dim):
            basis = np.zeros(len(x))
            basis[i] = h
            stencil = x + basis
            val = self.f(stencil)
            stencil_pts.append((stencil, val))
        return min(stencil_pts, key=lambda t:t[1])

    def run(self, x):
        f_val = self.f(x)
        noise = self.noise_f.estimate(x)
        grad, h = fd_gradient(f, x, noise)
        stencil_pt, stencil_val = self.get_stencil_pt(x, h)
        k = 0
        while not self.is_convergence(grad, f_val, 1e-2):
            d = self.lbfgs.calculate_direction(grad, k)
            new_pt, step, flag = self.ls.search((x, f_val, grad), d, noise)
            if not flag:
                new_pt, h, noise = self.rec.recover((x, f_val, grad), h, d, stencil_pt)
            else:
                h = None
            x_new, f_val_new = new_pt
            grad_new, h_new = fd_gradient(f, x_new, noise, h)
            stencil_pt, stencil_val = self.get_stencil_pt(x_new, h_new)
            s, y = x_new - x, grad_new - grad
            if self.curvature_satisfied(s, y):
                self.lbfgs.update_history(s, y)
            x, f_val = x_new, f_val_new
            k += 1

    def curvature_satisfied(self, s, y):
        return np.inner(s, y) >= self.zeta * norm(s) * norm(y)

    def is_convergence(self, grad, f_val, tol = 1e-8):
        return np.max(np.abs(grad)) <= tol

def f(x):
    return np.inner(x,x) * (1 + np.random.normal(0, 1e-4))

if __name__ == "__main__":
    fdlm = FDLM(f, (1e-8, 7, 100), (1e-2, 0.9, 10), (0.9, 1.1), 0.5, 10)
    fdlm.run(np.array([1, 1]))

