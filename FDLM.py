import numpy as np
from ECNoise import ECNoise
from FD import fd_gradient
from LBFGS import LBFGS
from linesearch import LineSearch
from numpy.linalg import norm
from recovery import Recovery
from numpy.core.umath_tests import inner1d

class FDLM(object):
    """
    The class of Finite Difference L-BFGS Method(FDLM)
    algorithm to minimize a noisy function
    """
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
        """
        Compute and store the best point on the stencil

        @param x: current point
        @param h: current finite-difference interval 

        @return: (x_s, f_s): best point on the stencil S = {x_i: x_i = x + h * e_i, i = 1,...,n} and its function value
        """
        stencil_pts = []
        dim = len(x)
        for i in range(dim): #evaluate the stencil points
            basis = np.zeros(len(x))
            basis[i] = h
            stencil = x + basis
            val = self.f(stencil)
            stencil_pts.append((stencil, val))
        return min(stencil_pts, key=lambda t:t[1])

    def run(self, x):
        """
        the FDLM method

        @param: x current iterate

        @print: current iteration, current iterate, current function value, current gradient
        """
        f_val = self.f(x) #evaluate the function value at initial iterate
        noise = self.noise_f.estimate(x) #estimate noise level at initial iterate
        grad, h = fd_gradient(f, x, noise) #compute finite difference interval and the corresponding finite gradient estimate
        stencil_pt, stencil_val = self.get_stencil_pt(x, h) #calculate the best stencil points and its function value
        k, fval_history = 0, [f_val] #set iteration counter and function value history
        while k < 150: #not self.is_convergence(f_val, fval_history, grad, 5, 1e-6): #while convergence test is not satisfied
        #while not self.is_convergence(grad, f_val, 1e-5):
            #print("k", k)
            d = self.lbfgs.calculate_direction(grad) #calculate LBFGS direction
            new_pt, step, flag = self.ls.search((x, f_val, grad), d, noise) #conduct linesearch to find the next iterate
            if not flag: #if linesearch failed 
                new_pt, h, noise = self.rec.recover((x, f_val, grad), h, d, (stencil_pt, stencil_val)) #call recovery mechanism
            x_new, f_val_new = new_pt 
            grad_new, _ = fd_gradient(f, x_new, noise, h) #calculate the finite-difference gradient estimator for the next iterate
            stencil_pt, stencil_val = self.get_stencil_pt(x_new, h) #calculate the new best stencil point
            s, y = x_new - x, grad_new - grad #calculate LBFGS parameter
            if self.curvature_satisfied(s, y): #if the curvature condition is satisfied
                self.lbfgs.update_history(s, y) #store new (s,y) pair
            x, f_val, grad = x_new, f_val_new, grad_new #update iterates
            fval_history.append(f_val) #append new function evaluations 
            print("Iter: {}, x: {}, fval: {}, grad: {}".format(k, x, f_val, grad)) 
            k += 1 #increase iteration counter

    def curvature_satisfied(self, s, y):
        """
        check the curvature condition: s'y >= zeta*norm(s)*norm(y)

        @param s: x_k+1 - x_k
        @param y: grad_k+1 - grad_k

        @return: 
            TRUE: curvature condition is satisfied
            FALSE: curvature condition is not satisfied
        """
        return np.inner(s, y) >= self.zeta * norm(s) * norm(y)

    def is_convergence(self, f_val, fval_history, grad, history_len = 5, tol = 1e-8):
        """
        convergence test for current iterate: terminates if either the moving average condition or gradient condition holds

        @param f_val: current iterate function value
        @param fval_history: stored function values for past history_len iterations
        @param grad: current gradient
        @param history_len: number of function values stored
        @tol: convergence tolerance for FDLM

        @return:
            TRUE: the convergence test is satisfied
            FALSE: the convergence test is not satisfied
        """
        if len(fval_history) <= 1:
            return np.max(np.abs(grad)) <= tol
        if len(fval_history) > history_len: #if more than history_len function values have been stored
            fval_history = fval_history[1:] #move the oldest one
        fval_ma = np.mean(fval_history) #calculate the moving average of length history_len
        #check moving average condition and gradient max norm condition; if either is met, the termination tolerance is met
        return np.max(np.abs(grad)) <= tol #or abs(f_val - fval_ma) <= tol * max(1, abs(fval_ma))

"""
test problem

def f(x):
    return np.inner(x,x) + x[0] + 1e-2*np.random.rand()#np.random.uniform(-1e-3,1e-3)
"""

def f(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2 

if __name__ == "__main__":
    fdlm = FDLM(f, (1e-8, 7, 100), (1e-4, 0.5, 20), (0.9, 1.1), 0.5, 10)
    fdlm.run(np.array([2, 2]))

