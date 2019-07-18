import math
import numpy as np
from np.linalg import norm
from FD import fd_gradient
from ECNoise import ECNoise

# Current TODO's:
#			Armijo relaxation,
#			Wolfe condition,
#			integration of ECNoise and gradient estimate
#			number of function evaluations
#			modularity
#			migrading recoverty to main FDLM

class LineSearch(object):
	def __init__(self, f, c1, c2, gamma_1, gamma_2, max_iter, mode = "fc"):
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
		self.gamma_1 = gamma_1
		self.gamma_2 = gamma_2
		self.max_iter = max_iter
		self.mode = mode #TODO migrate

	@staticmethod
	def is_armijo(orig_pt, fx_new, step, d, noise = 0.0):
		"""verify if the Armijo condition is satisfied
		@param orig_pt: tuple (fx, grad_fx): starting point at which we examine the Armijo condition
		@param fx_new:  f(x + step * d): ending point
		@param step:    step size
		@param d:       Armijo search direction
		"""
		fx_orig, grad_fx_orig = orig_pt
		return fx_new <= fx_orig + self.c1 * step * np.inner(grad_fx_orig, d) + 2 * noise

	@staticmethod
	def is_wolfe(orig_pt, x_new, d, noise = 0.0):
		fx_orig, grad_fx_orig = orig_pt
		grad_fx_new = fd_gradient(self.f, x_new, noise)
		return np.inner(grad_fx_new, d) >= self.c2 * np.inner(grad_fx_orig, d)

	def search(self, orig_pt, d):
		ls_counter = 0
		step = 1
		xk, f_xk, grad_f_xk = orig_pt

		# TODO ECNoise noise =
		while ls_counter < self.max_iter:
			relax = ls_counter >= 1
			xh = xk + step * d
			f_xh = self.f(xh)
			if LineSearch.is_armijo(orig_pt, f_xh, step, d, (ls_counter >= 1) * noise) and \
			   LineSearch.is_wolfe(orig_pt, xh, d, (ls_counter >= 1) * noise):
				return (xh, f_xh), step, TODO, True
			ls_counter += 1
			step /= 2
		return (xk, f_xk, grad_f_xk), step, TODO, False

	def recover(self, orig_pt, h, d, stencil_pt):
		#TODO migrate recovery to main class
		"""recover from failing line search
		@param orig_pt:    tuple (x, f(x), grad_f(x)): current point from which we recover failing line search
		@param h:     	   current finite-difference interval
		@param d: 	  	   current search direction
		@param stencil_pt: tuple (s, f(s)): best point on the stencil
		"""
		xk, f_xk, grad_f_xk = orig_pt
		xs, f_xs = stencil_pt
		rec_counter = 0
		#TODO: eps_d =
		#TODO: rec_counter +=
		#TODO: h_new =
		if h_new < self.gamma_1 * h or h_new > self.gamma_2 * h:
			h, x_out, f_out = h_new, xk, f_xk
		else:
			step = h / norm(d)
			xh = xk + step * d
			f_xh = f(xh)
			rec_counter += 1
			if LineSearch.is_armijo((f_xk, grad_f_xk), f_xh, step, d):
				x_out, f_out = xh, f_xh
			elif f_xh <= f_xs and f_xh <= f_xk:
				x_out, f_out = xh, f_xh
			elif f_xk > f_xs and f_xh > f_xs:
				x_out, f_out = stencil_pt
			else:
				x_out, f_out = xk, f_xk
				#TODO: eps_d
				#TODO: rec_counter +=
				#TODO: h_new =
		return x_out, f_out, h, rec_counter
