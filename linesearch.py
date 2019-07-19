import math
import numpy as np
from numpy.linalg import norm
from FD import fd_gradient
from ECNoise import ECNoise

# Current TODO's:
#			integration of ECNoise and gradient estimate
#			number of function evaluations
#			modularity

class LineSearch(object):
	def __init__(self, f, c1, c2, max_iter, mode = "fc"):
		"""
		@param f:  noisy function f to be evaluated
		@param c1: Armijo coefficient
		@param c2: Wolfe coefficient
		@param gamma_1:
		@param gamma_2:
		@param max_iter: maximum number of trials to perform line search
		"""
		self.f = f
		self.c1 = c1
		self.c2 = c2
		self.max_iter = max_iter
		self.mode = mode # TODO

	def is_armijo(self, orig_pt, fx_new, step, d, noise = 0.0):
		"""verify if the Armijo condition is satisfied
		@param orig_pt: tuple (fx, grad_fx): starting point at which we examine the Armijo condition
		@param fx_new:  f(x + step * d): ending point
		@param step:    step size
		@param d:       Armijo search direction
		"""
		x, fx_orig, grad_fx_orig = orig_pt
		return fx_new <= fx_orig + self.c1 * step * np.inner(grad_fx_orig, d) + 2 * noise

	def is_wolfe(self, orig_pt, x_new, d, noise = 0.0):
		x, fx_orig, grad_fx_orig = orig_pt
		grad_fx_new, _ = fd_gradient(self.f, x_new, noise)
		return np.inner(grad_fx_new, d) >= self.c2 * np.inner(grad_fx_orig, d)

	def search(self, orig_pt, d, noise = 0.0):
		ls_counter = 0
		step = 1
		xk, f_xk, grad_f_xk = orig_pt
		while ls_counter < self.max_iter:
			relax = ls_counter >= 1
			xh = xk + step * d
			f_xh = self.f(xh)
			if self.is_armijo(orig_pt, f_xh, step, d, (ls_counter >= 1) * noise) and \
			   self.is_wolfe(orig_pt, xh, d, (ls_counter >= 1) * noise):
				return (xh, f_xh), step, True
			ls_counter += 1
			step /= 2
		return (xk, f_xk, grad_f_xk), step, False