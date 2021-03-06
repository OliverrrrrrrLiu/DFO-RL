import numpy as np
from ECNoise import ECNoise
from FD import fd_gradient
from LBFGS import LBFGS
from linesearch import LineSearch
from numpy.linalg import norm
from Recovery_noise import Recovery
from numpy.core.umath_tests import inner1d
from LSfunc_def import *

###This code assumes that the noise level of the function is known!!

class FDLM(object):
    """
    The class of Finite Difference L-BFGS Method(FDLM)
    algorithm to minimize a noisy function
    """
    def __init__(self, noise, f, ecn_params, ls_params, rec_params, zeta, m, output_level, his_len = 5, mode = "fd", tol = 1e-6):
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
        self.rec = Recovery(f, *rec_params, noise, self.ls)
        self.lbfgs = LBFGS(m)
        self.output_level = output_level
        self.mode = mode
        self.tol = tol
        self.his_len = his_len
        self.noise = noise

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
        dim = len(x)
        print("Running from starting point:", x)
        f_val = self.f(x) #evaluate the function value at initial iterate
        self.eval_counter += 1
        grad, h = fd_gradient(self.f, x, self.noise, mode=self.mode) #compute finite difference interval and the corresponding finite gradient estimate
        if self.mode == "fd":
            self.eval_counter += dim
        else:
            self.eval_counter += 2 * dim
        norm_grad_k = np.max(np.abs(grad))
        stencil_pt, stencil_val = self.get_stencil_pt(x, h) #calculate the best stencil points and its function value
        k, fval_history = 0, [f_val] #set iteration counter and function value history
        eval_history = [self.eval_counter]
        num_func_it = 0
        num_func_evals = 0
        step = 0
        if self.output_level >= 2:
            output_header = '%6s %23s %9s %6s %9s' % \
            ('iter', 'f',  'alpha', '#func', '||grad_f||')
            print(output_header)
            print('%6i %23.16e  %9.2e %6i %9.2e' %
              (k, f_val, step, self.eval_counter, norm_grad_k))
        max_eval = 500 * dim

        while not self.is_convergence(f_val, fval_history, norm_grad_k,  self.tol, self.his_len) and self.eval_counter <= max_eval: #while convergence test is not satisfied
        #while not self.is_convergence(grad, f_val, 1e-5):
            #print("k", k)
            d = self.lbfgs.calculate_direction(grad) #calculate LBFGS direction
            #d = -grad
            new_pt, step, flag, ls_counter = self.ls.search((x, f_val, grad), d, noise=self.noise, mode=self.mode) #conduct linesearch to find the next iterate
            self.eval_counter += ls_counter
            if not flag: #if linesearch failed 
                print("linesearch failed")
                break
                #new_pt, h, noise, rec_counter = self.rec.recover((x, f_val, grad), h, d, (stencil_pt, stencil_val)) #call recovery mechanism
                #self.eval_counter += rec_counter
            x_new, f_val_new = new_pt 
            grad_new, _ = fd_gradient(self.f, x_new, self.noise, h, mode=self.mode) #calculate the finite-difference gradient estimator for the next iterate
            #print("true_grad: {}, est_grad: {}".format(self.get_grad(x_new), grad_new)) 
            if self.mode == "fd":
                self.eval_counter += dim
            else:
                self.eval_counter += 2 * dim           
            stencil_pt, stencil_val = self.get_stencil_pt(x_new, h) #calculate the new best stencil point
            s, y = x_new - x, grad_new - grad #calculate LBFGS parameter
            if self.curvature_satisfied(s, y): #if the curvature condition is satisfied
                self.lbfgs.update_history(s, y) #store new (s,y) pair
            x, f_val, grad = x_new, f_val_new, grad_new #update iterates
            norm_grad_k = np.max(np.abs(grad))
            fval_history.append(f_val) #append new function evaluations
            eval_history.append(self.eval_counter) 
            #if len(fval_history) > self.his_len: #history len
            #    fval_history = fval_history[1:]
            k += 1 #increase iteration counter
            if self.output_level >= 2:
                if k % 10 == 0:
                    print(output_header)
                print('%6i %23.16e  %9.2e %6i %9.2e' %
              (k, f_val, step, self.eval_counter, norm_grad_k))
            #print("true_grad: {}, est_grad: {}".format(self.get_grad(x), grad)) 
        stats = {}
        stats['num_iter'] = k
        stats['norm_grad'] = norm_grad_k
        stats['num_func_it'] = num_func_evals

        #Final output message
        if self.output_level >= 1:
            print('')
            print('Final objective.................: %g' % f_val)
            print('||grad|| at final point.........: %g' % norm_grad_k)
            print('Number of iterations............: %d' % k)
            print('Number of function evaluations..: %d' % self.eval_counter)
            print('')


        # Return output arguments
        print("Returned point:", x)
        return x, f_val, stats, fval_history, eval_history


    def curvature_satisfied(self, s, y):
        """
        check the curvature condition: s'y >= zeta*norm(s)*norm(y)

        @param s: x_k+1 - x_k
        @param y: grad_k+1 - grad_k

        @return: 
            TRUE: curvature condition is satisfied
            FALSE: curvature condition is not satisfied
        """
        #print(np.inner(s, y) >= self.zeta * norm(s) * norm(y))
        return np.inner(s, y) >= self.zeta * norm(s) * norm(y) and np.inner(s,y) > 0

    def is_convergence(self, f_val, fval_history, norm_grad, tol = 1e-6, history_len = 5):
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
            return norm_grad <= tol
        #if len(fval_history) > history_len: #if more than history_len function values have been stored
        #    fval_history = fval_history[1:] #move the oldest one
        fval_ma = np.mean(fval_history[-history_len:]) #calculate the moving average of length history_len
        #check moving average condition and gradient max norm condition; if either is met, the termination tolerance is met
        return abs(f_val - fval_ma) <= tol * max(1, abs(fval_ma)) or norm_grad <= tol

"""
test problem

def f(x):
    return np.inner(x,x) + x[0] + 1e-2*np.random.rand()#np.random.uniform(-1e-3,1e-3)
"""

def f(x):
    #return (np.inner(x,x))
    #return (np.inner(x,x) + x[0])*(1+ 1e-2*np.random.rand())#np.random.uniform(-1e-3,1e-3)
    #return (100*(x[1]-x[0]**2)**2 + (1-x[0])**2 ) * (1 + 1e-3 * np.random.rand()) 
    #return (100*(x[1]-x[0]**2 + 1e-3 * np.random.rand())**2 + (1-x[0] + 1e-3 * np.random.rand())**2 )
    #return x[1] + 0.00001 * (x[1] - x[0])**2

    # evaluate the Rosenbrock function
    _x = np.zeros(len(x))
    _x[:-1] = x[1:]
    res = np.sum(100 * (_x[:-1] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2) #*(1 + 1e-3 * np.random.rand())
    return res

    # The arwhead function
    #res = (x**2 + x[-1]**2)**2 - 4*x + 3
    #res = sum( res ) - res[-1]
    #return res + 1e-3 * np.random.rand()

    #booth function
    #return ((x[0]+2*x[1]-7)**2 + (2*x[0]+x[1]-5)**2) + 1e-2 * np.random.rand()
if __name__ == "__main__":
    fdlm = FDLM(0,f, (1e-8, 7, 100), (1e-4, 0.9, 100), (0.9, 1.1), 0, 10,3, his_len = 5, mode = "fd")
    x = np.array([-1.2,1]).astype(float)
    _, _, _, fval_history, eval_history = fdlm.run(x)

